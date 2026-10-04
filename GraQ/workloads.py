from __future__ import annotations

import numpy as np

from .circuit import Workload, parameter_id, pauli_coefficients, unitary_derivative


def make_case(spec: Workload, seed: int, gid_objective: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.45, size=spec.input_size)
    theta = rng.normal(scale=0.35, size=spec.parameters)
    if not gid_objective:
        return x, theta, rng.normal(scale=0.5, size=(spec.qubits, 3))
    # The objective direction is derived from the terminal local Euler block,
    # not from sampled gradients. For each qubit, Hry+Hrz is represented in
    # the local Pauli basis, yielding a certified commutator identity.
    cotangent = np.zeros((spec.qubits, 3), dtype=float)
    final_layer = spec.depth
    for qubit in range(spec.qubits):
        operators = []
        for axis in range(2):
            parameter = parameter_id(spec, final_layer, qubit, axis)
            unitary, derivative = unitary_derivative(spec, theta, parameter)
            operators.append(2j * derivative @ unitary.conj().T)
        local_sum = operators[0] + operators[1]
        cotangent[qubit] = pauli_coefficients(spec, local_sum)[qubit]
    scale = max(float(np.linalg.norm(cotangent)), 1e-12)
    return x, theta, cotangent / scale


def evaluation_workloads() -> list[tuple[Workload, bool]]:
    return [
        (Workload(4, 2, "block", "block_gid"), True),
        (Workload(4, 2, "brickwork", "brickwork_random"), False),
        (Workload(4, 2, "dense", "dense_negative"), False),
    ]
