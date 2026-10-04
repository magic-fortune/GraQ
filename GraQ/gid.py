from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import numpy as np

from .circuit import Workload, observable, unitary_derivative


@dataclass(frozen=True)
class IdentityCertificate:
    A: np.ndarray
    rank: int
    nullity: int
    residual: float
    source: str
    tolerance: float
    parameter_count: int | None = None
    scope_digest: str | None = None
    operator_scale: float = 0.0


def certificate_scope(spec: Workload, theta: np.ndarray, cotangent: np.ndarray) -> str:
    """Bind a certificate to the circuit, parameter point, and target VJP."""

    theta = np.ascontiguousarray(np.asarray(theta, dtype=np.float64).reshape(-1))
    cotangent = np.ascontiguousarray(
        np.asarray(cotangent, dtype=np.float64).reshape(spec.qubits, 3)
    )
    if theta.size != spec.parameters:
        raise ValueError(f"expected {spec.parameters} parameters, got {theta.size}")
    digest = hashlib.sha256()
    structure = (
        spec.qubits,
        spec.depth,
        spec.topology,
        spec.block_size,
        spec.seed,
        theta.shape,
        cotangent.shape,
    )
    digest.update(repr(structure).encode("ascii"))
    digest.update(theta.tobytes())
    digest.update(cotangent.tobytes())
    return digest.hexdigest()


def batch_certificate_scope(certificates: list[IdentityCertificate]) -> str:
    """Bind a common certificate to an ordered batch of sample certificates."""

    if not certificates or any(cert.scope_digest is None for cert in certificates):
        raise ValueError("every batch member must have a certificate scope")
    digest = hashlib.sha256(b"GraQ-common-batch-sum-v1")
    for certificate in certificates:
        member = certificate.scope_digest.encode("ascii")
        digest.update(len(member).to_bytes(4, "big"))
        digest.update(member)
    return digest.hexdigest()


def build_certificate(
    spec: Workload,
    theta: np.ndarray,
    cotangent: np.ndarray,
    tolerance: float = 1e-9,
) -> IdentityCertificate:
    if tolerance <= 0.0:
        raise ValueError("certificate tolerance must be positive")
    theta = np.asarray(theta, dtype=float).reshape(-1)
    cotangent = np.asarray(cotangent, dtype=float).reshape(spec.qubits, 3)
    if theta.size != spec.parameters:
        raise ValueError(f"expected {spec.parameters} parameters, got {theta.size}")

    operator = observable(spec, cotangent)
    columns = []
    for parameter in range(spec.parameters):
        unitary, derivative = unitary_derivative(spec, theta, parameter)
        generator = 2j * derivative @ unitary.conj().T
        k = 0.5j * (generator @ operator - operator @ generator)
        columns.append(k.reshape(-1))
    complex_matrix = np.column_stack(columns)
    # Gradient identities use real coefficients.  Solve the nullspace over
    # R by stacking the real and imaginary parts of the operator columns.
    matrix = np.vstack([complex_matrix.real, complex_matrix.imag])
    # For the usual tall operator matrix, an economic SVD retains every
    # right-singular vector while avoiding an O(rows**2) left factor.  If an
    # unusually deep circuit makes the matrix wide, full_matrices=True is
    # still required to recover the complete right nullspace.
    full_matrices = matrix.shape[0] < matrix.shape[1]
    _u, singular, vh = np.linalg.svd(matrix, full_matrices=full_matrices)
    operator_scale = float(singular[0]) if singular.size else 0.0
    numerical_floor = np.finfo(float).eps * max(matrix.shape)
    rank_threshold = max(tolerance, numerical_floor) * operator_scale
    rank = int(np.sum(singular > rank_threshold)) if operator_scale else 0
    if rank == matrix.shape[1]:
        A = np.zeros((0, matrix.shape[1]), dtype=float)
    else:
        A = np.real(vh[rank:, :])
        rows = np.linalg.norm(A, axis=1, keepdims=True)
        A = A / np.maximum(rows, 1e-15)
    absolute_residual = float(np.max(np.abs(complex_matrix @ A.T))) if A.size else 0.0
    residual = absolute_residual / operator_scale if operator_scale else absolute_residual
    return IdentityCertificate(
        A=A,
        rank=rank,
        nullity=A.shape[0],
        residual=residual,
        source="analytic-commutator-nullspace",
        tolerance=tolerance,
        parameter_count=spec.parameters,
        scope_digest=certificate_scope(spec, theta, cotangent),
        operator_scale=operator_scale,
    )


def certificate_rejection_reason(
    certificate: IdentityCertificate,
    parameter_count: int,
    *,
    expected_scope: str | None = None,
) -> str | None:
    a = np.asarray(certificate.A, dtype=float)
    if a.ndim != 2 or a.shape[1] != parameter_count:
        return "parameter_dimension_mismatch"
    if certificate.parameter_count not in (None, parameter_count):
        return "parameter_dimension_mismatch"
    if certificate.nullity != a.shape[0] or certificate.rank + certificate.nullity != parameter_count:
        return "certificate_metadata_mismatch"
    if not np.all(np.isfinite(a)) or not np.isfinite(certificate.residual):
        return "nonfinite_certificate"
    if certificate.residual > certificate.tolerance:
        return "certificate_residual_exceeded"
    if expected_scope is not None and certificate.scope_digest != expected_scope:
        return "certificate_scope_mismatch"
    if certificate.nullity and np.linalg.matrix_rank(a) != certificate.nullity:
        return "rank_deficient_certificate"
    return None


def decode_with_status(
    raw: np.ndarray,
    certificate: IdentityCertificate,
    *,
    expected_scope: str | None = None,
) -> tuple[np.ndarray, str | None]:
    raw = np.asarray(raw, dtype=float).reshape(-1)
    reason = certificate_rejection_reason(
        certificate,
        raw.size,
        expected_scope=expected_scope,
    )
    if reason is not None:
        return raw.copy(), reason
    if certificate.nullity == 0:
        return raw.copy(), "no_identity"
    if expected_scope is None:
        return raw.copy(), "certificate_scope_unverified"
    a = np.asarray(certificate.A, dtype=float)
    decoded = raw - a.T @ np.linalg.pinv(a @ a.T) @ (a @ raw)
    if not np.all(np.isfinite(decoded)):
        return raw.copy(), "nonfinite_decode"
    return decoded, None


def decode(
    raw: np.ndarray,
    certificate: IdentityCertificate,
    *,
    expected_scope: str | None = None,
) -> np.ndarray:
    decoded, _reason = decode_with_status(raw, certificate, expected_scope=expected_scope)
    return decoded


def constraint_residual(vector: np.ndarray, certificate: IdentityCertificate) -> float:
    if certificate.nullity == 0:
        return 0.0
    return float(np.linalg.norm(certificate.A @ np.asarray(vector, dtype=float)))


def certificates_compatible(certificates: list[IdentityCertificate], tolerance: float = 1e-9) -> bool:
    """Return whether all certificates define the same linear constraint space."""
    if not certificates:
        return False
    first = certificates[0]
    parameter_count = np.asarray(first.A).shape[1] if np.asarray(first.A).ndim == 2 else -1
    if parameter_count < 0 or any(
        certificate_rejection_reason(cert, parameter_count) is not None
        for cert in certificates
    ):
        return False
    if any(cert.nullity != first.nullity for cert in certificates[1:]):
        return False
    if first.nullity == 0:
        return True

    def projector(cert: IdentityCertificate) -> np.ndarray:
        a = cert.A
        return a.T @ np.linalg.pinv(a @ a.T) @ a

    reference = projector(first)
    return all(np.allclose(projector(cert), reference, atol=tolerance, rtol=tolerance)
               for cert in certificates[1:])


def common_certificate(certificates: list[IdentityCertificate], tolerance: float = 1e-9) -> IdentityCertificate | None:
    """Return a batch-safe certificate, or None when a common one is absent."""
    if not certificates_compatible(certificates, tolerance=tolerance):
        return None
    try:
        scope_digest = batch_certificate_scope(certificates)
    except ValueError:
        return None
    return replace(
        certificates[0],
        scope_digest=scope_digest,
        source=f"common-batch:{certificates[0].source}",
    )
