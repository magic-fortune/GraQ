from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .circuit import Workload, exact_gradient, shifted_states, state
from .measurement import merge_local_axes, prepare_local_measurement, sample_local_observables
from .cgr import compile_cgr
from .csf import CSFCompilation, causal_supports, compile_csf
from .gid import IdentityCertificate, build_certificate, certificate_scope, decode_with_status


METHODS = (
    "ps_xyz", "consumer_axis", "cgr", "csf", "cgr_csf", "gid",
    "cgr_gid", "csf_gid", "graq",
)


@dataclass
class Plan:
    spec: Workload
    x: np.ndarray
    theta: np.ndarray
    cotangent: np.ndarray
    supports: list[frozenset[int]]
    groups: list[tuple[int, ...]]
    consumer_groups: list[tuple[int, ...]]
    csf: CSFCompilation
    consumer_csf: CSFCompilation
    cgr_axes: list[np.ndarray]
    cgr_supported: int
    certificate: IdentityCertificate
    truth: np.ndarray


def compile_plan(spec: Workload, x: np.ndarray, theta: np.ndarray, cotangent: np.ndarray) -> Plan:
    supports = causal_supports(spec)
    cgr = compile_cgr(spec, x, theta, cotangent)
    csf = compile_csf(spec, cgr["axes"])
    consumer_csf = compile_csf(spec, np.asarray(cotangent, dtype=float))
    certificate = build_certificate(spec, theta, cotangent)
    truth = exact_gradient(spec, x, theta, cotangent)
    return Plan(spec, np.asarray(x, float), np.asarray(theta, float), np.asarray(cotangent, float),
                supports, list(csf.groups), list(consumer_csf.groups), csf, consumer_csf,
                cgr["axes"], cgr["supported"], certificate, truth)


def _masked_axis(plan: Plan, parameter: int, axis: np.ndarray) -> np.ndarray:
    support = plan.supports[parameter]
    masked = np.zeros_like(axis)
    masked[list(support)] = axis[list(support)]
    return masked


def _measure_pair(
    plus: np.ndarray,
    minus: np.ndarray,
    observables: list[np.ndarray],
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not observables:
        return np.zeros(0, dtype=float)
    setting = merge_local_axes(observables)
    if not np.any(np.linalg.norm(setting, axis=1) > 1e-12):
        return np.zeros(len(observables), dtype=float)
    plus_distribution = prepare_local_measurement(plus, setting)
    minus_distribution = prepare_local_measurement(minus, setting)
    plus_values = sample_local_observables(plus_distribution, np.asarray(observables), shots, rng)
    minus_values = sample_local_observables(minus_distribution, np.asarray(observables), shots, rng)
    return 0.5 * (plus_values - minus_values)


def _gradient_from_shifts(
    plan: Plan,
    parameter: int,
    axis: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    masked = _masked_axis(plan, parameter, axis)
    if not np.any(np.linalg.norm(masked, axis=1) > 1e-12):
        return 0.0, 0
    plus, minus = shifted_states(plan.spec, plan.x, plan.theta, parameter)
    return float(_measure_pair(plus, minus, [masked], shots, rng)[0]), 2


def estimate(plan: Plan, method: str, shots: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    if method not in METHODS:
        raise ValueError(method)
    gradient = np.zeros(plan.spec.parameters)
    circuits = 0
    if method in ("ps_xyz", "gid"):
        for parameter in range(plan.spec.parameters):
            plus, minus = shifted_states(plan.spec, plan.x, plan.theta, parameter)
            # All qubits can be measured in the same Pauli basis setting, so
            # XYZ parameter-shift costs three settings per sign, not one
            # setting per qubit and axis.
            for axis in range(3):
                setting = np.zeros((plan.spec.qubits, 3)); setting[:, axis] = plan.cotangent[:, axis]
                setting = _masked_axis(plan, parameter, setting)
                if not np.any(np.linalg.norm(setting, axis=1) > 1e-12):
                    continue
                gradient[parameter] += _measure_pair(plus, minus, [setting], shots, rng)[0]
                circuits += 2
    elif method == "consumer_axis":
        for parameter in range(plan.spec.parameters):
            gradient[parameter], used = _gradient_from_shifts(
                plan, parameter, plan.cotangent, shots, rng
            )
            circuits += used
    elif method in ("cgr", "cgr_gid"):
        for parameter, axis in enumerate(plan.cgr_axes):
            gradient[parameter], used = _gradient_from_shifts(plan, parameter, axis, shots, rng)
            circuits += used
    else:
        use_cgr = method in ("cgr_csf", "graq")
        axes = plan.cgr_axes if use_cgr else [plan.cotangent] * plan.spec.parameters
        groups = plan.groups if use_cgr else plan.consumer_groups
        for group in groups:
            plus_theta = plan.theta.copy(); minus_theta = plan.theta.copy()
            for parameter in group:
                plus_theta[parameter] += np.pi / 2
                minus_theta[parameter] -= np.pi / 2
            plus = state(plan.spec, plan.x, plus_theta)
            minus = state(plan.spec, plan.x, minus_theta)
            active_parameters = []
            observables = []
            for parameter in group:
                masked = _masked_axis(plan, parameter, axes[parameter])
                if np.any(np.linalg.norm(masked, axis=1) > 1e-12):
                    active_parameters.append(parameter)
                    observables.append(masked)
            values = _measure_pair(plus, minus, observables, shots, rng)
            for parameter, value in zip(active_parameters, values):
                gradient[parameter] = value
            if not observables:
                continue
            circuits += 2
    if method in ("gid", "cgr_gid", "csf_gid", "graq"):
        decoded, fallback_reason = decode_with_status(
            gradient,
            plan.certificate,
            expected_scope=certificate_scope(plan.spec, plan.theta, plan.cotangent),
        )
        return decoded, {
            "circuits": circuits,
            "shots": circuits * shots,
            "jobs": circuits,
            "raw": gradient,
            "gid_applied": fallback_reason is None,
            "gid_fallback_reason": fallback_reason,
        }
    return gradient, {"circuits": circuits, "shots": circuits * shots, "jobs": circuits, "raw": gradient}
