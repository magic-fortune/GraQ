from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .circuit import (
    PAULIS,
    Workload,
    _embed_one,
    exact_gradient,
    input_state,
    outputs_from_state,
    shifted_states,
    state,
    unitary,
)
from .engine import Plan, estimate
from .io import write_json
from .measurement import (
    local_observable_moments,
    prepare_local_measurement,
    sample_local_observables,
    sample_local_xyz,
)
from .cgr import compile_cgr
from .csf import causal_supports, compile_csf
from .gid import IdentityCertificate, build_certificate


TRAINING_METHODS = (
    "ideal_ps",
    "frozen_q",
    "ps_xyz",
    "ps_sgd",
    "qoc",
    "scaled_ps",
    "gcans",
    "consumer_axis",
    "cgr",
    "cgr_csf",
    "graq",
)

EXTERNAL_METHODS = (
    "ideal_ps",
    "ps_xyz",
    "ps_sgd",
    "qoc",
    "scaled_ps",
    "gcans",
    "graq",
)

INTERNAL_METHODS = (
    "ps_xyz",
    "consumer_axis",
    "cgr",
    "cgr_csf",
    "graq",
)

METHOD_LABELS = {
    "ideal_ps": "Ideal PS",
    "frozen_q": "Frozen-Q",
    "ps_xyz": "PS-XYZ",
    "ps_sgd": "PS-SGD",
    "qoc": "QOC",
    "scaled_ps": "Scaled-PS",
    "gcans": "gCANS",
    "consumer_axis": "Consumer-axis",
    "cgr": "CGR",
    "cgr_csf": "CGR+CSF",
    "graq": "GraQ",
}


@dataclass
class Adam:
    shape: tuple[int, ...]
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        self.m = np.zeros(self.shape, dtype=float)
        self.v = np.zeros(self.shape, dtype=float)
        self.step = 0

    def update(self, value: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self.step += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * gradient * gradient
        m_hat = self.m / (1.0 - self.beta1 ** self.step)
        v_hat = self.v / (1.0 - self.beta2 ** self.step)
        return value - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent)


def _sample_outputs(
    statevector: np.ndarray,
    qubits: int,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return sample_local_xyz(statevector, qubits, shots, rng)


def _sample_weighted(
    statevector: np.ndarray,
    axis: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    distribution = prepare_local_measurement(statevector, axis)
    estimate = float(sample_local_observables(distribution, axis, shots, rng)[0])
    variance = float(local_observable_moments(distribution, axis)[1][0, 0])
    return estimate, variance


def _sample_consumer_observable(
    statevector: np.ndarray,
    cotangent: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> float:
    distribution = prepare_local_measurement(statevector, cotangent)
    return float(sample_local_observables(distribution, cotangent, shots, rng)[0])


def _input_gradient(
    spec: Workload,
    angles: np.ndarray,
    theta: np.ndarray,
    cotangent: np.ndarray,
    shots: int,
    rng: np.random.Generator,
    exact: bool,
) -> tuple[np.ndarray, dict]:
    gradient = np.zeros(spec.input_size, dtype=float)
    for parameter in range(spec.input_size):
        plus = angles.copy()
        minus = angles.copy()
        plus[parameter] += np.pi / 2
        minus[parameter] -= np.pi / 2
        plus_state = state(spec, plus, theta)
        minus_state = state(spec, minus, theta)
        if exact:
            plus_output = outputs_from_state(plus_state, spec.qubits)
            minus_output = outputs_from_state(minus_state, spec.qubits)
            gradient[parameter] = 0.5 * float(np.sum((plus_output - minus_output) * cotangent))
        else:
            plus_value = _sample_consumer_observable(plus_state, cotangent, shots, rng)
            minus_value = _sample_consumer_observable(minus_state, cotangent, shots, rng)
            gradient[parameter] = 0.5 * (plus_value - minus_value)
    if exact:
        return gradient, {"circuits": 0, "shots": 0, "jobs": 0}
    circuits = 2 * spec.input_size
    return gradient, {"circuits": circuits, "shots": circuits * shots, "jobs": circuits}


def _empty_certificate(parameters: int) -> IdentityCertificate:
    return IdentityCertificate(
        A=np.zeros((0, parameters), dtype=float),
        rank=parameters,
        nullity=0,
        residual=0.0,
        source="not-requested",
        tolerance=1e-9,
    )


def _make_plan(
    spec: Workload,
    x: np.ndarray,
    theta: np.ndarray,
    cotangent: np.ndarray,
    supports: list[frozenset[int]],
    certificate: bool,
) -> Plan:
    cgr = compile_cgr(spec, x, theta, cotangent)
    csf = compile_csf(spec, cgr["axes"])
    consumer_csf = compile_csf(spec, np.asarray(cotangent, dtype=float))
    cert = build_certificate(spec, theta, cotangent) if certificate else _empty_certificate(spec.parameters)
    truth = exact_gradient(spec, x, theta, cotangent) if certificate else np.zeros(spec.parameters)
    return Plan(
        spec=spec,
        x=np.asarray(x, dtype=float),
        theta=np.asarray(theta, dtype=float),
        cotangent=np.asarray(cotangent, dtype=float),
        supports=supports,
        groups=list(csf.groups),
        consumer_groups=list(consumer_csf.groups),
        csf=csf,
        consumer_csf=consumer_csf,
        cgr_axes=cgr["axes"],
        cgr_supported=cgr["supported"],
        certificate=cert,
        truth=truth,
    )


def _estimate_ps_subset(
    plan: Plan,
    parameters: np.ndarray,
    shots_per_parameter: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    gradient = np.zeros(plan.spec.parameters, dtype=float)
    single_shot_std = np.zeros(plan.spec.parameters, dtype=float)
    circuits = 0
    total_shots = 0
    for parameter in parameters:
        parameter = int(parameter)
        shots = int(shots_per_parameter[parameter])
        plus, minus = shifted_states(plan.spec, plan.x, plan.theta, parameter)
        support = plan.supports[parameter]
        component_variance = 0.0
        for axis in range(3):
            observable = np.zeros((plan.spec.qubits, 3), dtype=float)
            indices = list(support)
            observable[indices, axis] = plan.cotangent[indices, axis]
            plus_value, plus_variance = _sample_weighted(plus, observable, shots, rng)
            minus_value, minus_variance = _sample_weighted(minus, observable, shots, rng)
            gradient[parameter] += 0.5 * (plus_value - minus_value)
            component_variance += 0.25 * (plus_variance + minus_variance)
            circuits += 2
            total_shots += 2 * shots
        single_shot_std[parameter] = math.sqrt(max(component_variance, 0.0))
    return gradient, {
        "circuits": circuits,
        "shots": total_shots,
        "jobs": circuits,
        "single_shot_std": single_shot_std,
        "raw": gradient.copy(),
    }


def _batch_observables(spec: Workload) -> list[np.ndarray]:
    return [
        _embed_one(PAULIS[axis], qubit, spec.qubits)
        for qubit in range(spec.qubits)
        for axis in range(3)
    ]


def _encode_angles(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    preactivation = features @ weight.T + bias
    return np.pi * np.tanh(preactivation / np.pi)


def _encoded_states(spec: Workload, angles: np.ndarray) -> np.ndarray:
    return np.asarray([input_state(spec, row) for row in angles], dtype=complex)


def _batch_outputs(
    spec: Workload,
    states: np.ndarray,
    theta: np.ndarray,
    observables: list[np.ndarray],
) -> np.ndarray:
    transformed = (unitary(spec, theta) @ states.T).T
    values = [
        np.real(np.einsum("bi,ij,bj->b", transformed.conj(), operator, transformed))
        for operator in observables
    ]
    return np.column_stack(values).reshape(len(states), spec.qubits, 3)


def _accuracy(
    spec: Workload,
    features: np.ndarray,
    theta: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    adapter_weight: np.ndarray,
    adapter_bias: np.ndarray,
    observables: list[np.ndarray],
    labels: np.ndarray,
) -> float:
    angles = _encode_angles(features, adapter_weight, adapter_bias)
    states = _encoded_states(spec, angles)
    features = _batch_outputs(spec, states, theta, observables).reshape(len(states), -1)
    predictions = np.argmax(features @ weight.T + bias, axis=1)
    return float(np.mean(predictions == labels))


def _training_order(size: int, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order: list[int] = []
    while len(order) < steps:
        order.extend(rng.permutation(size).tolist())
    return np.asarray(order[:steps], dtype=int)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _classical_context(
    train_x: np.ndarray,
    train_y: np.ndarray,
    final_x: np.ndarray,
    final_y: np.ndarray,
    feature_source: str,
) -> dict:
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=2000, C=1.0, random_state=20260922)
        model.fit(train_x, train_y)
        return {
            "method": "input_feature_linear",
            "feature_source": feature_source,
            "input_dimensions": int(train_x.shape[1]),
            "final_accuracy": float(model.score(final_x, final_y)),
            "quantum_circuits": 0,
            "scope": "application-context baseline; not a quantum execution competitor",
        }
    except ImportError:
        return {"method": "frozen_bert_pca_linear", "available": False}


def _paired_interval(
    rows: list[dict], left: str, right: str, metric: str = "final_accuracy"
) -> dict:
    left_rows = {row["seed"]: row for row in rows if row["method"] == left}
    right_rows = {row["seed"]: row for row in rows if row["method"] == right}
    seeds = sorted(set(left_rows) & set(right_rows))
    differences = np.asarray(
        [right_rows[seed][metric] - left_rows[seed][metric] for seed in seeds],
        dtype=float,
    )
    if not len(differences):
        return {
            "left": left,
            "right": right,
            "metric": metric,
            "paired_seeds": 0,
            "mean_difference": None,
            "ci95": None,
        }
    if len(differences) <= 1:
        half_width = 0.0
    else:
        half_width = 2.776 * float(np.std(differences, ddof=1)) / math.sqrt(len(differences))
    mean = float(np.mean(differences))
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "paired_seeds": len(seeds),
        "mean_difference": mean,
        "ci95": [mean - half_width, mean + half_width],
    }


def run_one(
    dataset: str,
    method: str,
    seed: int,
    arrays: np.lib.npyio.NpzFile,
    array_prefix: str,
    feature_source: str,
    steps: int,
    derivative_shots: int,
    forward_shots: int,
    threshold: float,
    checkpoint: int,
    quantum_optimizer: str,
    quantum_learning_rate: float,
) -> dict:
    if method not in TRAINING_METHODS:
        raise ValueError(method)
    if quantum_optimizer not in ("method_default", "adam", "sgd"):
        raise ValueError(f"unknown quantum optimizer: {quantum_optimizer}")
    if quantum_learning_rate <= 0.0:
        raise ValueError("quantum learning rate must be positive")
    resolved_optimizer = quantum_optimizer
    if resolved_optimizer == "method_default":
        resolved_optimizer = "sgd" if method in ("ps_sgd", "gcans") else "adam"

    spec = Workload(4, 2, "block", f"{dataset}_4q_d2", block_size=4, seed=0)
    train_x = np.asarray(arrays[f"{array_prefix}_train_x"], dtype=float)
    validation_x = np.asarray(arrays[f"{array_prefix}_validation_x"], dtype=float)
    final_x = np.asarray(arrays[f"{array_prefix}_final_x"], dtype=float)
    train_y = np.asarray(arrays[f"{array_prefix}_train_y"], dtype=int)
    validation_y = np.asarray(arrays[f"{array_prefix}_validation_y"], dtype=int)
    final_y = np.asarray(arrays[f"{array_prefix}_final_y"], dtype=int)
    classes = int(np.max(train_y) + 1)

    initializer = np.random.default_rng(seed)
    theta = initializer.normal(scale=0.15, size=spec.parameters)
    weight = initializer.normal(scale=0.08, size=(classes, 3 * spec.qubits))
    bias = np.zeros(classes, dtype=float)
    adapter_weight = initializer.normal(scale=0.01, size=(spec.input_size, train_x.shape[1]))
    adapter_weight[:, :spec.input_size] += 0.25 * np.eye(spec.input_size)
    adapter_bias = np.zeros(spec.input_size, dtype=float)
    initial_theta = theta.copy()
    initial_weight = weight.copy()
    initial_adapter = adapter_weight.copy()
    order = _training_order(len(train_x), steps, seed + 17)
    forward_rng = np.random.default_rng(seed + 1000)
    adapter_rng = np.random.default_rng(seed + 1500)
    derivative_rng = np.random.default_rng(seed + 2000)
    selection_rng = np.random.default_rng(seed + 3000)

    theta_adam = Adam(theta.shape, learning_rate=quantum_learning_rate)
    weight_adam = Adam(weight.shape, learning_rate=0.03)
    bias_adam = Adam(bias.shape, learning_rate=0.03)
    adapter_weight_adam = Adam(adapter_weight.shape, learning_rate=0.015)
    adapter_bias_adam = Adam(adapter_bias.shape, learning_rate=0.015)
    supports = causal_supports(spec)
    causal_groups = 0

    observables = _batch_observables(spec)
    circuits = 0
    shots = 0
    jobs = 0
    gid_steps = 0
    gid_nullity = 0
    gid_raw_squared_error = 0.0
    gid_decoded_squared_error = 0.0
    qoc_magnitude = np.ones(spec.parameters, dtype=float)
    gcans_gradient = np.zeros(spec.parameters, dtype=float)
    gcans_std = np.ones(spec.parameters, dtype=float)
    gcans_shots = np.full(spec.parameters, 32, dtype=int)
    curve: list[dict] = []
    started = time.perf_counter()
    best_validation_accuracy = -np.inf
    best_validation_step = 0
    best_validation_circuits = 0
    best_validation_shots = 0
    best_validation_seconds = 0.0
    best_theta = theta.copy()
    best_weight = weight.copy()
    best_bias = bias.copy()
    best_adapter_weight = adapter_weight.copy()
    best_adapter_bias = adapter_bias.copy()

    def record(step: int) -> None:
        nonlocal best_validation_accuracy
        nonlocal best_validation_step, best_validation_circuits
        nonlocal best_validation_shots, best_validation_seconds
        nonlocal best_theta, best_weight, best_bias
        nonlocal best_adapter_weight, best_adapter_bias
        accuracy = _accuracy(
            spec,
            validation_x,
            theta,
            weight,
            bias,
            adapter_weight,
            adapter_bias,
            observables,
            validation_y,
        )
        curve.append(
            {
                "step": step,
                "validation_accuracy": accuracy,
                "circuits": circuits,
                "shots": shots,
                "jobs": jobs,
                "seconds": time.perf_counter() - started,
            }
        )
        if accuracy > best_validation_accuracy:
            best_validation_accuracy = accuracy
            best_validation_step = step
            best_validation_circuits = circuits
            best_validation_shots = shots
            best_validation_seconds = time.perf_counter() - started
            best_theta = theta.copy()
            best_weight = weight.copy()
            best_bias = bias.copy()
            best_adapter_weight = adapter_weight.copy()
            best_adapter_bias = adapter_bias.copy()

    record(0)
    for step, sample_id in enumerate(order, start=1):
        raw_x = train_x[sample_id]
        label = int(train_y[sample_id])
        preactivation = adapter_weight @ raw_x + adapter_bias
        x = np.pi * np.tanh(preactivation / np.pi)
        current_state = state(spec, x, theta)
        exact_output = outputs_from_state(current_state, spec.qubits)
        if method == "ideal_ps":
            noisy_output = exact_output
        else:
            noisy_output = _sample_outputs(current_state, spec.qubits, forward_shots, forward_rng)
            circuits += 3
            shots += 3 * forward_shots
            jobs += 3

        feature = noisy_output.reshape(-1)
        probability = _softmax(weight @ feature + bias)
        downstream = probability.copy()
        downstream[label] -= 1.0
        cotangent = (weight.T @ downstream).reshape(spec.qubits, 3)
        weight_gradient = np.outer(downstream, feature) + 1e-4 * weight
        bias_gradient = downstream

        input_gradient, input_resource = _input_gradient(
            spec=spec,
            angles=x,
            theta=theta,
            cotangent=cotangent,
            shots=derivative_shots,
            rng=adapter_rng,
            exact=method == "ideal_ps",
        )
        circuits += int(input_resource["circuits"])
        shots += int(input_resource["shots"])
        jobs += int(input_resource["jobs"])
        activation_gradient = input_gradient * (1.0 - np.tanh(preactivation / np.pi) ** 2)
        adapter_weight_gradient = np.outer(activation_gradient, raw_x) + 1e-4 * adapter_weight
        adapter_bias_gradient = activation_gradient

        if method == "frozen_q":
            quantum_gradient = np.zeros(spec.parameters, dtype=float)
            resource = {"circuits": 0, "shots": 0, "jobs": 0}
        elif method == "ideal_ps":
            quantum_gradient = exact_gradient(spec, x, theta, cotangent)
            resource = {"circuits": 0, "shots": 0, "jobs": 0}
        else:
            need_certificate = method == "graq"
            plan = _make_plan(spec, x, theta, cotangent, supports, need_certificate)
            causal_groups = len(plan.groups)
            if method in ("ps_xyz", "ps_sgd", "scaled_ps"):
                parameters = np.arange(spec.parameters)
                shot_vector = np.full(spec.parameters, derivative_shots, dtype=int)
                quantum_gradient, resource = _estimate_ps_subset(plan, parameters, shot_vector, derivative_rng)
                if method == "scaled_ps":
                    dimension = 2 ** spec.qubits
                    copies = 2 * derivative_shots
                    scale = dimension * copies / (2 * dimension * dimension + dimension * copies - 2)
                    quantum_gradient = scale * quantum_gradient
                    resource["scale"] = float(scale)
            elif method == "qoc":
                phase = (step - 1) % 3
                if phase == 0:
                    parameters = np.arange(spec.parameters)
                else:
                    keep = int(math.ceil(0.5 * spec.parameters))
                    probability_vector = qoc_magnitude / max(float(np.sum(qoc_magnitude)), 1e-12)
                    parameters = np.sort(
                        selection_rng.choice(spec.parameters, size=keep, replace=False, p=probability_vector)
                    )
                shot_vector = np.full(spec.parameters, derivative_shots, dtype=int)
                quantum_gradient, resource = _estimate_ps_subset(plan, parameters, shot_vector, derivative_rng)
                if phase == 0:
                    qoc_magnitude = np.maximum(np.abs(quantum_gradient), 1e-12)
                resource["selected_parameters"] = int(len(parameters))
            elif method == "gcans":
                parameters = np.arange(spec.parameters)
                quantum_gradient, resource = _estimate_ps_subset(plan, parameters, gcans_shots, derivative_rng)
                smoothing = 0.9
                gcans_gradient = smoothing * gcans_gradient + (1.0 - smoothing) * quantum_gradient
                observed_std = np.asarray(resource["single_shot_std"], dtype=float)
                gcans_std = smoothing * gcans_std + (1.0 - smoothing) * observed_std
                learning_rate = 0.025
                lipschitz = 10.0
                factor = 2.0 * lipschitz * learning_rate / (2.0 - lipschitz * learning_rate)
                denominator = max(float(np.dot(gcans_gradient, gcans_gradient)), 1e-8)
                proposed = np.ceil(factor * gcans_std * np.sum(gcans_std) / denominator)
                gcans_shots = np.clip(proposed, 16, 512).astype(int)
                resource["next_shots_min"] = int(np.min(gcans_shots))
                resource["next_shots_max"] = int(np.max(gcans_shots))
            else:
                engine_method = {
                    "consumer_axis": "consumer_axis",
                    "cgr": "cgr",
                    "cgr_csf": "cgr_csf",
                    "graq": "graq",
                }[method]
                quantum_gradient, resource = estimate(plan, engine_method, derivative_shots, derivative_rng)
                if method == "graq":
                    raw = np.asarray(resource["raw"], dtype=float)
                    gid_steps += int(plan.certificate.nullity > 0)
                    gid_nullity += int(plan.certificate.nullity)
                    gid_raw_squared_error += float(np.sum((raw - plan.truth) ** 2))
                    gid_decoded_squared_error += float(np.sum((quantum_gradient - plan.truth) ** 2))

        circuits += int(resource["circuits"])
        shots += int(resource["shots"])
        jobs += int(resource["jobs"])

        if method not in ("frozen_q",):
            if resolved_optimizer == "sgd":
                theta = theta - quantum_learning_rate * quantum_gradient
            else:
                theta = theta_adam.update(theta, quantum_gradient)
        weight = weight_adam.update(weight, weight_gradient)
        bias = bias_adam.update(bias, bias_gradient)
        adapter_weight = adapter_weight_adam.update(adapter_weight, adapter_weight_gradient)
        adapter_bias = adapter_bias_adam.update(adapter_bias, adapter_bias_gradient)

        if step % checkpoint == 0 or step == steps:
            record(step)

    final_accuracy = _accuracy(
        spec,
        final_x,
        theta,
        weight,
        bias,
        adapter_weight,
        adapter_bias,
        observables,
        final_y,
    )
    final_accuracy_at_best_validation = _accuracy(
        spec,
        final_x,
        best_theta,
        best_weight,
        best_bias,
        best_adapter_weight,
        best_adapter_bias,
        observables,
        final_y,
    )
    reached = next((point for point in curve if point["validation_accuracy"] >= threshold), None)
    return {
        "dataset": dataset,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "quantum_optimizer": resolved_optimizer,
        "quantum_learning_rate": quantum_learning_rate,
        "seed": seed,
        "steps": steps,
        "qubits": spec.qubits,
        "depth": spec.depth,
        "parameters": spec.parameters,
        "causal_groups": causal_groups,
        "array_prefix": array_prefix,
        "train_examples": int(len(train_x)),
        "validation_examples": int(len(validation_x)),
        "final_examples": int(len(final_x)),
        "input_dimensions_used": int(train_x.shape[1]),
        "feature_source": feature_source,
        "input_projection": (
            f"trainable {train_x.shape[1]}-to-{spec.input_size} angle adapter; "
            "no feature bypass"
        ),
        "batch_size": 1,
        "initialization_hash": hashlib.sha256(
            initial_theta.tobytes() + initial_weight.tobytes() + initial_adapter.tobytes()
        ).hexdigest(),
        "final_accuracy": final_accuracy,
        "final_accuracy_at_best_validation": final_accuracy_at_best_validation,
        "best_validation_accuracy": float(best_validation_accuracy),
        "best_validation_step": best_validation_step,
        "best_validation_circuits": best_validation_circuits,
        "best_validation_shots": best_validation_shots,
        "best_validation_seconds": best_validation_seconds,
        "threshold": threshold,
        "reached": reached is not None,
        "time_to_quality": reached,
        "circuits": circuits,
        "shots": shots,
        "jobs": jobs,
        "seconds": time.perf_counter() - started,
        "gid_steps": gid_steps,
        "mean_gid_nullity": gid_nullity / steps if method == "graq" else 0.0,
        "mean_gid_raw_squared_error": (
            gid_raw_squared_error / max(gid_steps, 1) if method == "graq" else None
        ),
        "mean_gid_decoded_squared_error": (
            gid_decoded_squared_error / max(gid_steps, 1) if method == "graq" else None
        ),
        "curve": curve,
    }


def run_experiment(
    dataset: str,
    data_path: Path,
    output_path: Path,
    array_prefix: str | None = None,
    feature_source: str = "frozen BERT CLS plus train-only whitened PCA snapshot",
    manifest_path: Path | None = None,
    steps: int = 512,
    seeds: tuple[int, ...] = (73001, 73002, 73003, 73004, 73005),
    methods: tuple[str, ...] = TRAINING_METHODS,
    derivative_shots: int = 128,
    forward_shots: int = 256,
    threshold: float | None = None,
    checkpoint: int = 64,
    quantum_optimizer: str = "method_default",
    quantum_learning_rate: float = 0.025,
) -> dict:
    thresholds = {"sst2": 0.60, "ag_news": 0.50, "mnist": 0.60, "fashion_mnist": 0.45}
    threshold = thresholds[dataset] if threshold is None else threshold
    array_prefix = dataset if array_prefix is None else array_prefix
    arrays = np.load(data_path)
    required = [
        f"{array_prefix}_{split}_{suffix}"
        for split in ("train", "validation", "final")
        for suffix in ("x", "y")
    ]
    missing = [name for name in required if name not in arrays.files]
    if missing:
        raise KeyError(f"missing arrays for {array_prefix}: {missing}")
    train_x = np.asarray(arrays[f"{array_prefix}_train_x"], dtype=float)
    train_y = np.asarray(arrays[f"{array_prefix}_train_y"], dtype=int)
    final_x = np.asarray(arrays[f"{array_prefix}_final_x"], dtype=float)
    final_y = np.asarray(arrays[f"{array_prefix}_final_y"], dtype=int)
    manifest = json.loads(manifest_path.read_text()) if manifest_path is not None else None
    rows: list[dict] = []
    partial_path = output_path.with_name(f"{output_path.stem}_partial.json")
    for seed_index, seed in enumerate(seeds):
        rotation = seed_index % len(methods)
        ordered_methods = methods[rotation:] + methods[:rotation]
        for method in ordered_methods:
            row = run_one(
                dataset=dataset,
                method=method,
                seed=seed,
                arrays=arrays,
                array_prefix=array_prefix,
                feature_source=feature_source,
                steps=steps,
                derivative_shots=derivative_shots,
                forward_shots=forward_shots,
                threshold=threshold,
                checkpoint=checkpoint,
                quantum_optimizer=quantum_optimizer,
                quantum_learning_rate=quantum_learning_rate,
            )
            rows.append(row)
            write_json(partial_path, {"dataset": dataset, "rows": rows})
            print(
                dataset,
                method,
                seed,
                f"accuracy={row['final_accuracy']:.4f}",
                f"circuits={row['circuits']}",
                flush=True,
            )

    summary: dict[str, dict] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        final = np.asarray([row["final_accuracy"] for row in selected], dtype=float)
        reached = [row for row in selected if row["reached"]]
        summary[method] = {
            "label": METHOD_LABELS[method],
            "runs": len(selected),
            "mean_final_accuracy": float(np.mean(final)),
            "stdev_final_accuracy": float(np.std(final, ddof=1)) if len(final) > 1 else 0.0,
            "mean_best_validation_accuracy": float(np.mean([row["best_validation_accuracy"] for row in selected])),
            "mean_final_accuracy_at_best_validation": float(
                np.mean([row["final_accuracy_at_best_validation"] for row in selected])
            ),
            "stdev_final_accuracy_at_best_validation": float(
                np.std([row["final_accuracy_at_best_validation"] for row in selected], ddof=1)
            ) if len(selected) > 1 else 0.0,
            "mean_best_validation_step": float(np.mean([row["best_validation_step"] for row in selected])),
            "mean_best_validation_circuits": float(
                np.mean([row["best_validation_circuits"] for row in selected])
            ),
            "mean_best_validation_shots": float(
                np.mean([row["best_validation_shots"] for row in selected])
            ),
            "reached_rate": float(len(reached) / len(selected)),
            "mean_circuits": float(np.mean([row["circuits"] for row in selected])),
            "mean_shots": float(np.mean([row["shots"] for row in selected])),
            "mean_jobs": float(np.mean([row["jobs"] for row in selected])),
            "mean_seconds": float(np.mean([row["seconds"] for row in selected])),
            "mean_circuits_to_quality": float(np.mean([row["time_to_quality"]["circuits"] for row in reached])) if reached else None,
            "mean_shots_to_quality": float(np.mean([row["time_to_quality"]["shots"] for row in reached])) if reached else None,
            "mean_gid_raw_squared_error": (
                float(np.mean([row["mean_gid_raw_squared_error"] for row in selected]))
                if method == "graq" else None
            ),
            "mean_gid_decoded_squared_error": (
                float(np.mean([row["mean_gid_decoded_squared_error"] for row in selected]))
                if method == "graq" else None
            ),
        }

    full_rows = [row for row in rows if row["method"] == "graq"]
    payload = {
        "protocol": {
            "date": date.today().isoformat(),
            "dataset": dataset,
            "array_prefix": array_prefix,
            "steps": steps,
            "seeds": list(seeds),
            "methods": list(methods),
            "quantum_optimizer": quantum_optimizer,
            "quantum_learning_rate": quantum_learning_rate,
            "selection_policy": (
                "select checkpoint by maximum validation accuracy; evaluate final split once "
                "at the selected checkpoint"
            ),
            "gid_error_metric": (
                "mean per-step squared L2 norm across quantum parameters; raw and decoded "
                "errors are paired on the same GraQ estimate; unavailable for other methods"
            ),
            "external_methods": list(EXTERNAL_METHODS),
            "internal_methods": list(INTERNAL_METHODS),
            "derivative_shots": derivative_shots,
            "forward_shots": forward_shots,
            "finite_shot_protocol_version": 2,
            "sampling_model": "joint bitstrings per product-basis setting with cross-qubit covariance",
            "ps_support_policy": "causal-support-matched to CGR",
            "batch_size": 1,
            "threshold": threshold,
            "feature_source": feature_source,
            "feature_dimensions_available": int(train_x.shape[1]),
            "feature_dimensions_used": int(train_x.shape[1]),
            "train_examples": int(len(train_x)),
            "validation_examples": int(len(arrays[f"{array_prefix}_validation_y"])),
            "final_examples": int(len(final_y)),
            "data_path": str(data_path),
            "data_sha256": _file_sha256(data_path),
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "manifest_sha256": _file_sha256(manifest_path) if manifest_path is not None else None,
            "dataset_repository": manifest.get("repository") if manifest is not None else None,
            "dataset_revision": manifest.get("revision") if manifest is not None else None,
            "trainable_input_adapter": True,
            "input_adapter_gradient": "parameter shift over eight encoded angles; common cost for all finite-shot methods",
            "classical_input_bypass": False,
            "qoc": {"accumulation_window": 1, "pruning_window": 2, "pruning_ratio": 0.5},
            "scaled_ps": "lambda=d*N/(2*d^2+d*N-2), d=2^qubits, N=2*derivative_shots",
            "gcans": {"minimum_shots": 16, "maximum_shots": 512, "lipschitz": 10.0, "learning_rate": 0.025},
        },
        "context": {
            "classical_linear": _classical_context(
                train_x,
                train_y,
                final_x,
                final_y,
                feature_source,
            ),
            "gid_batch_scope": "batch size one; every certificate applies to the submitted logical gradient",
        },
        "summary": summary,
        "paired_comparisons": {
            "graq_minus_ps": _paired_interval(rows, "ps_xyz", "graq"),
            "graq_minus_cgr_csf": _paired_interval(rows, "cgr_csf", "graq"),
            "graq_minus_ps_at_best_validation": _paired_interval(
                rows, "ps_xyz", "graq", "final_accuracy_at_best_validation"
            ),
            "graq_minus_cgr_csf_at_best_validation": _paired_interval(
                rows, "cgr_csf", "graq", "final_accuracy_at_best_validation"
            ),
            "gcans_minus_ps_sgd": _paired_interval(rows, "ps_sgd", "gcans"),
        },
        "gid": {
            "runs": len(full_rows),
            "mean_active_steps": float(np.mean([row["gid_steps"] for row in full_rows])),
            "mean_nullity": float(np.mean([row["mean_gid_nullity"] for row in full_rows])),
        },
        "rows": rows,
    }
    write_json(output_path, payload)
    return payload
