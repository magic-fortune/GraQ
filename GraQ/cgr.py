from __future__ import annotations

import numpy as np

from .circuit import Workload


def _terminal_null_axis(spec: Workload, theta: np.ndarray, parameter: int) -> np.ndarray | None:
    """Return a certified null axis for the supported terminal RY/RZ fragment.

    Bulk-layer parameters deliberately return None and fall back to the raw
    consumer axis; this implementation is not a general tangent-space solver.
    """
    per_layer = 2 * spec.qubits
    layer, offset = divmod(parameter, per_layer)
    if layer != spec.depth:
        return None
    axis = offset % 2
    if axis == 1:  # terminal RZ: the Z component commutes with the observable
        return np.array([0.0, 0.0, 1.0])
    # terminal RY is followed by the same-qubit terminal RZ.
    qubit = offset // 2
    rz_parameter = parameter + 1
    phi = float(theta[rz_parameter])
    del qubit
    return np.array([-np.sin(phi), np.cos(phi), 0.0])


def project_consumer_axis(
    spec: Workload,
    x: np.ndarray,
    theta: np.ndarray,
    cotangent: np.ndarray,
    parameter: int,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, bool, float]:
    del x
    raw = np.asarray(cotangent, dtype=float).reshape(spec.qubits, 3)
    null_axis = _terminal_null_axis(spec, theta, parameter)
    if null_axis is None:
        return raw.copy(), False, 0.0
    component = np.sum(raw * null_axis[None, :], axis=-1, keepdims=True)
    projected = raw.copy()
    # Only the local output of this terminal parameter is constrained.
    qubit = (parameter % (2 * spec.qubits)) // 2
    projected[qubit] -= component[qubit, 0] * null_axis
    removed = float(np.linalg.norm(raw - projected))
    if removed <= tolerance:
        return raw.copy(), True, removed
    return projected, True, removed


def compile_cgr(spec: Workload, x: np.ndarray, theta: np.ndarray, cotangent: np.ndarray) -> dict:
    """Compile CGR axes, certifying only the circuit's terminal RY/RZ layer."""
    axes = []
    supported = 0
    removed = []
    for parameter in range(spec.parameters):
        axis, is_supported, removed_norm = project_consumer_axis(spec, x, theta, cotangent, parameter)
        axes.append(axis)
        supported += int(is_supported)
        removed.append(removed_norm)
    return {"axes": axes, "supported": supported, "removed_norms": removed}
