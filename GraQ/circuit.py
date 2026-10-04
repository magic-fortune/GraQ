from __future__ import annotations

from dataclasses import dataclass
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
PAULIS = (X, Y, Z)


@dataclass(frozen=True)
class Workload:
    """Small executable circuit family used by the fresh evaluation.

    Each circuit has two trainable rotations per qubit in every bulk layer and
    one terminal RY/RZ pair. The terminal pair is where the current CGR can
    extract a structural generator-null certificate; no artificial layer is
    added by the compiler.
    """

    qubits: int
    depth: int
    topology: str = "block"
    name: str = "workload"
    block_size: int = 4
    seed: int = 0

    @property
    def parameters(self) -> int:
        return 2 * self.qubits * (self.depth + 1)

    @property
    def input_size(self) -> int:
        return 2 * self.qubits


def parameter_id(spec: Workload, layer: int, qubit: int, axis: int) -> int:
    if axis not in (0, 1):
        raise ValueError("the executable circuit uses RY/RZ parameter pairs")
    return (layer * spec.qubits + qubit) * 2 + axis


def _rotation(axis: int, angle: float) -> np.ndarray:
    pauli = PAULIS[axis]
    return np.cos(angle / 2) * I2 - 1j * np.sin(angle / 2) * pauli


def _embed_one(gate: np.ndarray, qubit: int, qubits: int) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for index in range(qubits):
        result = np.kron(result, gate if index == qubit else I2)
    return result


def _apply_one_state(state: np.ndarray, gate: np.ndarray, qubit: int, qubits: int) -> np.ndarray:
    tensor = np.asarray(state, dtype=complex).reshape((2,) * qubits)
    updated = np.tensordot(gate, tensor, axes=([1], [qubit]))
    return np.moveaxis(updated, 0, qubit).reshape(-1)


def _apply_cnot_state(state: np.ndarray, control: int, target: int, qubits: int) -> np.ndarray:
    indices = np.arange(2 ** qubits)
    control_mask = 1 << (qubits - 1 - control)
    target_mask = 1 << (qubits - 1 - target)
    destinations = indices.copy()
    active = (indices & control_mask) != 0
    destinations[active] ^= target_mask
    result = np.empty_like(state)
    result[destinations] = state
    return result


def _cnot(control: int, target: int, qubits: int) -> np.ndarray:
    size = 2 ** qubits
    matrix = np.zeros((size, size), dtype=complex)
    for basis in range(size):
        bits = [(basis >> (qubits - 1 - i)) & 1 for i in range(qubits)]
        output = bits[:]
        if bits[control]:
            output[target] ^= 1
        index = 0
        for bit in output:
            index = (index << 1) | bit
        matrix[index, basis] = 1.0
    return matrix


def entangler_edges(spec: Workload, layer: int) -> list[tuple[int, int]]:
    topology = "separable" if spec.topology == "none" else spec.topology
    if topology == "separable":
        return []
    if topology == "block":
        edges: list[tuple[int, int]] = []
        for start in range(0, spec.qubits, spec.block_size):
            end = min(start + spec.block_size, spec.qubits)
            offset = (layer + spec.seed) % 2
            edges.extend((q, q + 1) for q in range(start + offset, end - 1, 2))
        return edges
    if topology in ("brickwork", "ring"):
        offset = (layer + spec.seed) % 2
        edges = [(q, q + 1) for q in range(offset, spec.qubits - 1, 2)]
        if topology == "ring" and (layer + spec.seed) % 2 and spec.qubits > 2:
            edges.append((spec.qubits - 1, 0))
        return edges
    if topology == "dense":
        center = (layer + spec.seed) % spec.qubits
        return [(center, target) for target in range(spec.qubits) if target != center]
    raise ValueError(f"unknown topology: {spec.topology}")


def _operations(spec: Workload, theta: np.ndarray) -> list[tuple[str, int | None, int | None, np.ndarray]]:
    operations: list[tuple[str, int | None, int | None, np.ndarray]] = []
    for layer in range(spec.depth):
        for qubit in range(spec.qubits):
            for axis in (0, 1):
                parameter = parameter_id(spec, layer, qubit, axis)
                gate = _rotation(axis + 1, float(theta[parameter]))
                operations.append(("rotation", qubit, parameter, gate))
        for control, target in entangler_edges(spec, layer):
            operations.append(("cnot", (control, target), None, _cnot(control, target, spec.qubits)))
    terminal = spec.depth
    for qubit in range(spec.qubits):
        for axis in (0, 1):
            parameter = parameter_id(spec, terminal, qubit, axis)
            gate = _rotation(axis + 1, float(theta[parameter]))
            operations.append(("rotation", qubit, parameter, gate))
    return operations


def input_state(spec: Workload, x: np.ndarray) -> np.ndarray:
    state = np.zeros(2 ** spec.qubits, dtype=complex)
    state[0] = 1.0
    for qubit in range(spec.qubits):
        state = _apply_one_state(state, _rotation(1, float(x[qubit])), qubit, spec.qubits)
        state = _apply_one_state(
            state, _rotation(2, float(x[spec.qubits + qubit])), qubit, spec.qubits
        )
    return state


def unitary(spec: Workload, theta: np.ndarray) -> np.ndarray:
    result = np.eye(2 ** spec.qubits, dtype=complex)
    for kind, wires, _parameter, gate in _operations(spec, np.asarray(theta, dtype=float)):
        if kind == "rotation":
            result = _embed_one(gate, int(wires), spec.qubits) @ result
        else:
            result = gate @ result
    return result


def unitary_derivative(spec: Workload, theta: np.ndarray, parameter: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    result = np.eye(2 ** spec.qubits, dtype=complex)
    derivative = np.zeros_like(result)
    found = False
    for kind, wires, pid, gate in _operations(spec, theta):
        if kind == "cnot":
            derivative = gate @ derivative
            result = gate @ result
            continue
        full_gate = _embed_one(gate, int(wires), spec.qubits)
        derivative = full_gate @ derivative
        if pid == parameter:
            axis = 0 if parameter % 2 == 0 else 1
            pauli = PAULIS[axis + 1]
            derivative = derivative + _embed_one((-0.5j * pauli) @ gate, int(wires), spec.qubits) @ result
            found = True
        result = full_gate @ result
    if not found:
        raise IndexError(parameter)
    return result, derivative


def state(spec: Workload, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    psi = input_state(spec, x)
    for layer in range(spec.depth):
        for qubit in range(spec.qubits):
            for axis in (0, 1):
                parameter = parameter_id(spec, layer, qubit, axis)
                psi = _apply_one_state(
                    psi, _rotation(axis + 1, float(theta[parameter])), qubit, spec.qubits
                )
        for control, target in entangler_edges(spec, layer):
            psi = _apply_cnot_state(psi, control, target, spec.qubits)
    terminal = spec.depth
    for qubit in range(spec.qubits):
        for axis in (0, 1):
            parameter = parameter_id(spec, terminal, qubit, axis)
            psi = _apply_one_state(
                psi, _rotation(axis + 1, float(theta[parameter])), qubit, spec.qubits
            )
    return psi


def observable(spec: Workload, vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(spec.qubits, 3)
    result = np.zeros((2 ** spec.qubits, 2 ** spec.qubits), dtype=complex)
    for qubit in range(spec.qubits):
        for axis in range(3):
            result += vector[qubit, axis] * _embed_one(PAULIS[axis], qubit, spec.qubits)
    return result


def outputs_from_state(psi: np.ndarray, qubits: int) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    if psi.size != 2 ** qubits:
        raise ValueError("state size does not match qubit count")
    values = []
    for qubit in range(qubits):
        for pauli in PAULIS:
            transformed = _apply_one_state(psi, pauli, qubit, qubits)
            values.append(float(np.real(np.vdot(psi, transformed))))
    return np.asarray(values).reshape(qubits, 3)


def outputs(spec: Workload, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return outputs_from_state(state(spec, x, theta), spec.qubits)


def shifted_states(
    spec: Workload,
    x: np.ndarray,
    theta: np.ndarray,
    parameter: int,
) -> tuple[np.ndarray, np.ndarray]:
    plus = np.asarray(theta, dtype=float).copy()
    minus = np.asarray(theta, dtype=float).copy()
    plus[parameter] += np.pi / 2
    minus[parameter] -= np.pi / 2
    return state(spec, x, plus), state(spec, x, minus)


def shifted_outputs(spec: Workload, x: np.ndarray, theta: np.ndarray, parameter: int) -> tuple[np.ndarray, np.ndarray]:
    plus, minus = shifted_states(spec, x, theta, parameter)
    return outputs_from_state(plus, spec.qubits), outputs_from_state(minus, spec.qubits)


def exact_gradient(spec: Workload, x: np.ndarray, theta: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
    gradient = np.zeros(spec.parameters)
    for parameter in range(spec.parameters):
        plus, minus = shifted_outputs(spec, x, theta, parameter)
        gradient[parameter] = 0.5 * float(np.sum((plus - minus) * cotangent))
    return gradient


def causal_supports(spec: Workload) -> list[frozenset[int]]:
    # Causal analysis is a compiler pass. Keep it matrix-free so large
    # workloads do not allocate 2**qubits-by-2**qubits CNOT matrices.
    structure: list[tuple[str, int | tuple[int, int] | None, int | None]] = []
    for layer in range(spec.depth):
        for qubit in range(spec.qubits):
            for axis in (0, 1):
                structure.append(("rotation", qubit, parameter_id(spec, layer, qubit, axis)))
        for control, target in entangler_edges(spec, layer):
            structure.append(("cnot", (control, target), None))
    terminal = spec.depth
    for qubit in range(spec.qubits):
        for axis in (0, 1):
            structure.append(("rotation", qubit, parameter_id(spec, terminal, qubit, axis)))
    # At every point, future_outputs[q] is the set of output wires reachable
    # from wire q through later gates. A reverse pass computes every parameter
    # cone without the old parameters x circuit-size nested scan.
    future_outputs = [frozenset((qubit,)) for qubit in range(spec.qubits)]
    supports: list[frozenset[int] | None] = [None] * spec.parameters
    for kind, wires, parameter in reversed(structure):
        if kind == "rotation":
            if parameter is not None:
                supports[parameter] = future_outputs[int(wires)]
            continue
        control, target = wires  # type: ignore[misc]
        merged = future_outputs[control] | future_outputs[target]
        future_outputs[control] = merged
        future_outputs[target] = merged
    if any(support is None for support in supports):
        missing = [index for index, support in enumerate(supports) if support is None]
        raise RuntimeError(f"support extraction missed parameters: {missing}")
    return [support for support in supports if support is not None]


def pauli_coefficients(spec: Workload, operator: np.ndarray) -> np.ndarray:
    coefficients = np.zeros((spec.qubits, 3), dtype=float)
    for qubit in range(spec.qubits):
        for axis, pauli in enumerate(PAULIS):
            embedded = _embed_one(pauli, qubit, spec.qubits)
            coefficients[qubit, axis] = float(np.real(np.trace(embedded.conj().T @ operator)) / (2 ** spec.qubits))
    return coefficients
