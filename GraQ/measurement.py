from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuit import _apply_one_state, _rotation


TOLERANCE = 1e-12


@dataclass(frozen=True)
class LocalMeasurementDistribution:
    """Joint bitstring distribution for one product-basis measurement setting."""

    probabilities: np.ndarray
    cdf: np.ndarray
    setting_units: np.ndarray

    @property
    def qubits(self) -> int:
        return int(self.setting_units.shape[0])


def merge_local_axes(observables: list[np.ndarray], tolerance: float = TOLERANCE) -> np.ndarray:
    if not observables:
        raise ValueError("at least one observable is required")
    arrays = [np.asarray(axis, dtype=float) for axis in observables]
    setting = np.zeros_like(arrays[0])
    for axis in arrays:
        if axis.shape != setting.shape or axis.ndim != 2 or axis.shape[1] != 3:
            raise ValueError("local observables must all have shape (qubits, 3)")
        for qubit, local in enumerate(axis):
            norm = float(np.linalg.norm(local))
            if norm <= tolerance:
                continue
            current = setting[qubit]
            current_norm = float(np.linalg.norm(current))
            if current_norm <= tolerance:
                setting[qubit] = local / norm
                continue
            denominator = max(current_norm * norm, tolerance)
            if float(np.linalg.norm(np.cross(current, local))) > tolerance * denominator:
                raise ValueError(f"incompatible local axes on qubit {qubit}")
    return setting


def prepare_local_measurement(
    statevector: np.ndarray,
    setting_axes: np.ndarray,
    tolerance: float = TOLERANCE,
) -> LocalMeasurementDistribution:
    axes = np.asarray(setting_axes, dtype=float)
    if axes.ndim != 2 or axes.shape[1] != 3:
        raise ValueError("setting_axes must have shape (qubits, 3)")
    qubits = int(axes.shape[0])
    rotated = np.asarray(statevector, dtype=complex).reshape(-1).copy()
    if rotated.size != 2 ** qubits:
        raise ValueError("state size does not match measurement axes")

    units = np.zeros_like(axes)
    for qubit, local in enumerate(axes):
        norm = float(np.linalg.norm(local))
        if norm <= tolerance:
            continue
        unit = local / norm
        units[qubit] = unit
        polar = float(np.arccos(np.clip(unit[2], -1.0, 1.0)))
        azimuth = float(np.arctan2(unit[1], unit[0]))
        if abs(azimuth) > tolerance:
            rotated = _apply_one_state(rotated, _rotation(2, -azimuth), qubit, qubits)
        if abs(polar) > tolerance:
            rotated = _apply_one_state(rotated, _rotation(1, -polar), qubit, qubits)

    probabilities = np.abs(rotated) ** 2
    probabilities /= float(np.sum(probabilities))
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    return LocalMeasurementDistribution(probabilities, cdf, units)


def _observable_coefficients(
    distribution: LocalMeasurementDistribution,
    observables: np.ndarray,
    tolerance: float = TOLERANCE,
) -> np.ndarray:
    axes = np.asarray(observables, dtype=float)
    if axes.ndim == 2:
        axes = axes[None, :, :]
    expected = (distribution.qubits, 3)
    if axes.ndim != 3 or axes.shape[1:] != expected:
        raise ValueError(f"observables must have shape {expected} or (count, {expected[0]}, 3)")

    coefficients = np.zeros((axes.shape[0], distribution.qubits), dtype=float)
    for observable, axis in enumerate(axes):
        for qubit, local in enumerate(axis):
            norm = float(np.linalg.norm(local))
            if norm <= tolerance:
                continue
            unit = distribution.setting_units[qubit]
            if float(np.linalg.norm(unit)) <= tolerance:
                raise ValueError(f"observable is active on unmeasured qubit {qubit}")
            coefficient = float(np.dot(local, unit))
            if float(np.linalg.norm(local - coefficient * unit)) > tolerance * max(norm, 1.0):
                raise ValueError(f"observable is incompatible with setting on qubit {qubit}")
            coefficients[observable, qubit] = coefficient
    return coefficients


def _outcomes(indices: np.ndarray, coefficients: np.ndarray, qubits: int) -> np.ndarray:
    shifts = np.arange(qubits - 1, -1, -1, dtype=np.int64)
    signs = 1.0 - 2.0 * ((indices[:, None] >> shifts[None, :]) & 1)
    return signs @ coefficients.T


def sample_local_observables(
    distribution: LocalMeasurementDistribution,
    observables: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    coefficients = _observable_coefficients(distribution, observables)
    if shots <= 0:
        return np.zeros(coefficients.shape[0], dtype=float)
    indices = np.searchsorted(distribution.cdf, rng.random(shots), side="right")
    return np.mean(_outcomes(indices, coefficients, distribution.qubits), axis=0)


def local_observable_moments(
    distribution: LocalMeasurementDistribution,
    observables: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = _observable_coefficients(distribution, observables)
    indices = np.arange(distribution.probabilities.size, dtype=np.int64)
    outcomes = _outcomes(indices, coefficients, distribution.qubits)
    means = distribution.probabilities @ outcomes
    centered = outcomes - means[None, :]
    covariance = centered.T @ (centered * distribution.probabilities[:, None])
    return means, covariance


def sample_local_observable(
    statevector: np.ndarray,
    observable: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> float:
    axis = np.asarray(observable, dtype=float)
    distribution = prepare_local_measurement(statevector, axis)
    return float(sample_local_observables(distribution, axis, shots, rng)[0])


def sample_local_xyz(
    statevector: np.ndarray,
    qubits: int,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = np.zeros((qubits, 3), dtype=float)
    for pauli_axis in range(3):
        setting = np.zeros((qubits, 3), dtype=float)
        setting[:, pauli_axis] = 1.0
        observables = np.zeros((qubits, qubits, 3), dtype=float)
        observables[np.arange(qubits), np.arange(qubits), pauli_axis] = 1.0
        distribution = prepare_local_measurement(statevector, setting)
        sampled[:, pauli_axis] = sample_local_observables(
            distribution, observables, shots, rng
        )
    return sampled
