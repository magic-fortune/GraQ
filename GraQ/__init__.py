"""GraQ: quantum gradient compilation, execution, and identity decoding."""

from .circuit import (
    Workload, exact_gradient, outputs, outputs_from_state, pauli_coefficients,
    shifted_outputs, shifted_states, state, unitary_derivative,
)
from .engine import METHODS, Plan, compile_plan, estimate
from .measurement import (
    LocalMeasurementDistribution, local_observable_moments, merge_local_axes,
    prepare_local_measurement, sample_local_observable, sample_local_observables,
    sample_local_xyz,
)
from .cgr import compile_cgr, project_consumer_axis
from .csf import (
    CSFCompilation, ParameterPlan, ShiftRule, build_parameter_plans, causal_supports,
    compile_csf, compile_parameter_plans, conflict_graph, conflict_reasons, consumer_conflict_graph,
    greedy_groups, validate_compilation, validate_groups,
)
from .gid import (
    IdentityCertificate, batch_certificate_scope, build_certificate, certificate_rejection_reason, certificate_scope,
    certificates_compatible, common_certificate, constraint_residual, decode, decode_with_status,
)
from .workloads import evaluation_workloads, make_case

__all__ = [
    "Workload", "Plan", "METHODS", "compile_plan", "estimate", "exact_gradient", "outputs",
    "outputs_from_state", "shifted_outputs", "shifted_states", "state", "unitary_derivative",
    "pauli_coefficients", "compile_cgr", "project_consumer_axis", "LocalMeasurementDistribution",
    "local_observable_moments", "merge_local_axes", "prepare_local_measurement",
    "sample_local_observable", "sample_local_observables", "sample_local_xyz",
    "CSFCompilation", "ParameterPlan", "ShiftRule", "build_parameter_plans", "causal_supports",
    "compile_csf", "compile_parameter_plans", "conflict_graph", "conflict_reasons", "consumer_conflict_graph",
    "greedy_groups", "validate_compilation", "validate_groups", "IdentityCertificate",
    "batch_certificate_scope", "build_certificate", "certificate_rejection_reason", "certificate_scope",
    "certificates_compatible", "common_certificate", "constraint_residual", "decode", "decode_with_status",
    "evaluation_workloads", "make_case",
]
