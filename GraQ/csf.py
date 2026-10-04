from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Hashable, Sequence

import numpy as np

from .circuit import Workload, causal_supports as _causal_supports


@dataclass(frozen=True)
class ShiftRule:
    plus: float = math.pi / 2
    minus: float = -math.pi / 2
    coefficient: float = 0.5


@dataclass(frozen=True)
class ParameterPlan:
    parameter: int
    influence: frozenset[int]
    consumer: frozenset[int]
    readout_axes: tuple[tuple[float, float, float], ...]
    physical_readout: tuple[Hashable, ...]
    shift_rule: ShiftRule
    physical_parameters: tuple[Hashable, ...]
    decode_key: Hashable
    batch_scope: Hashable
    barrier_scope: Hashable


@dataclass(frozen=True)
class CSFCompilation:
    parameters: tuple[ParameterPlan, ...]
    groups: tuple[tuple[int, ...], ...]
    conflicts: dict[int, frozenset[int]]
    reason_counts: dict[str, int]
    skipped_parameters: tuple[int, ...]

    @property
    def active_parameters(self) -> int:
        return len(self.parameters) - len(self.skipped_parameters)

    @property
    def compression(self) -> float:
        return self.active_parameters / len(self.groups) if self.groups else 1.0


def causal_supports(spec: Workload) -> list[frozenset[int]]:
    """Return parameter influence cones using the same DAG pass for all topologies."""

    return _causal_supports(spec)


def causal_supports_for_workload(spec: Workload) -> list[frozenset[int]]:
    return causal_supports(spec)


def _as_parameter_axes(
    spec: Workload,
    readout_axes: np.ndarray | Sequence[np.ndarray],
) -> list[np.ndarray]:
    if isinstance(readout_axes, np.ndarray) and readout_axes.shape == (spec.qubits, 3):
        return [np.asarray(readout_axes, dtype=float) for _ in range(spec.parameters)]
    axes = [np.asarray(axis, dtype=float).reshape(spec.qubits, 3) for axis in readout_axes]
    if len(axes) != spec.parameters:
        raise ValueError(f"expected {spec.parameters} parameter readouts, got {len(axes)}")
    return axes


def build_parameter_plans(
    spec: Workload,
    readout_axes: np.ndarray | Sequence[np.ndarray],
    *,
    shift_rules: Sequence[ShiftRule] | None = None,
    physical_parameter_map: Sequence[Sequence[Hashable]] | None = None,
    physical_readout_map: Sequence[Hashable] | None = None,
    decode_keys: Sequence[Hashable] | None = None,
    batch_scopes: Sequence[Hashable] | None = None,
    barrier_scopes: Sequence[Hashable] | None = None,
    tolerance: float = 1e-12,
) -> tuple[ParameterPlan, ...]:
    influences = causal_supports(spec)
    axes = _as_parameter_axes(spec, readout_axes)
    shift_rules = list([ShiftRule()] * spec.parameters if shift_rules is None else shift_rules)
    physical_parameter_map = list(
        [(p,) for p in range(spec.parameters)]
        if physical_parameter_map is None else physical_parameter_map
    )
    physical_readout_map = tuple(
        range(spec.qubits) if physical_readout_map is None else physical_readout_map
    )
    decode_keys = list(range(spec.parameters) if decode_keys is None else decode_keys)
    batch_scopes = list(
        ["default-batch"] * spec.parameters if batch_scopes is None else batch_scopes
    )
    barrier_scopes = list(
        ["default-barrier"] * spec.parameters if barrier_scopes is None else barrier_scopes
    )
    fields = {
        "shift_rules": shift_rules,
        "physical_parameter_map": physical_parameter_map,
        "decode_keys": decode_keys,
        "batch_scopes": batch_scopes,
        "barrier_scopes": barrier_scopes,
    }
    for name, values in fields.items():
        if len(values) != spec.parameters:
            raise ValueError(f"{name} has {len(values)} entries; expected {spec.parameters}")
    if len(physical_readout_map) != spec.qubits:
        raise ValueError("physical_readout_map must map every logical output qubit")

    plans = []
    for parameter, (influence, axis) in enumerate(zip(influences, axes)):
        active_outputs = {
            qubit for qubit in influence if float(np.linalg.norm(axis[qubit])) > tolerance
        }
        plans.append(ParameterPlan(
            parameter=parameter,
            influence=influence,
            consumer=frozenset(active_outputs),
            readout_axes=tuple(tuple(float(value) for value in row) for row in axis),
            physical_readout=physical_readout_map,
            shift_rule=shift_rules[parameter],
            physical_parameters=tuple(physical_parameter_map[parameter]),
            decode_key=decode_keys[parameter],
            batch_scope=batch_scopes[parameter],
            barrier_scope=barrier_scopes[parameter],
        ))
    return tuple(plans)


def _readout_compatible(left: ParameterPlan, right: ParameterPlan, tolerance: float = 1e-10) -> bool:
    # With an identical one-to-one layout, logically disjoint consumers are
    # physically disjoint too. conflict_reasons checks causal overlap first.
    if (
        left.physical_readout == right.physical_readout
        and len(set(left.physical_readout)) == len(left.physical_readout)
    ):
        return True
    left_axes = {
        left.physical_readout[q]: np.asarray(left.readout_axes[q], dtype=float)
        for q in left.consumer
    }
    right_axes = {
        right.physical_readout[q]: np.asarray(right.readout_axes[q], dtype=float)
        for q in right.consumer
    }
    for physical_qubit in left_axes.keys() & right_axes.keys():
        left_axis = left_axes[physical_qubit]
        right_axis = right_axes[physical_qubit]
        denominator = max(float(np.linalg.norm(left_axis) * np.linalg.norm(right_axis)), tolerance)
        if float(np.linalg.norm(np.cross(left_axis, right_axis))) > tolerance * denominator:
            return False
    return True


def conflict_reasons(left: ParameterPlan, right: ParameterPlan) -> tuple[str, ...]:
    """Return conservative reasons preventing fusion; causal overlap short-circuits."""

    reasons = []
    causal_overlap = bool(
        (right.influence & left.consumer) or (left.influence & right.consumer)
    )
    if causal_overlap:
        # Causal overlap is already sufficient to separate the parameters.
        # Compatibility checks cannot make this pair fusible, so avoid their
        # array work on the dominant path of large conflict graphs.
        return ("causal_overlap",)
    if not _readout_compatible(left, right):
        reasons.append("readout_incompatible")
    if left.shift_rule != right.shift_rule:
        reasons.append("shift_rule_incompatible")
    if set(left.physical_parameters) & set(right.physical_parameters):
        reasons.append("parameter_mapping_overlap")
    if left.decode_key == right.decode_key:
        reasons.append("decode_ambiguous")
    if left.batch_scope != right.batch_scope:
        reasons.append("batch_scope_mismatch")
    if left.barrier_scope != right.barrier_scope:
        reasons.append("barrier_scope_mismatch")
    return tuple(reasons)


def consumer_conflict_graph(
    parameters: Sequence[ParameterPlan],
) -> tuple[dict[int, set[int]], Counter[str]]:
    active = [parameter for parameter in parameters if parameter.consumer]
    graph = {parameter.parameter: set() for parameter in active}
    reason_counts: Counter[str] = Counter()
    for left_index, left in enumerate(active):
        for right in active[:left_index]:
            reasons = conflict_reasons(left, right)
            if not reasons:
                continue
            graph[left.parameter].add(right.parameter)
            graph[right.parameter].add(left.parameter)
            reason_counts.update(reasons)
    return graph, reason_counts


def _color_graph(graph: dict[int, set[int]]) -> tuple[tuple[int, ...], ...]:
    order = sorted(graph, key=lambda p: (-len(graph[p]), p))
    colors: dict[int, int] = {}
    for parameter in order:
        unavailable = {colors[neighbor] for neighbor in graph[parameter] if neighbor in colors}
        color = 0
        while color in unavailable:
            color += 1
        colors[parameter] = color
    groups: list[list[int]] = [[] for _ in range(max(colors.values(), default=-1) + 1)]
    for parameter, color in colors.items():
        groups[color].append(parameter)
    return tuple(tuple(sorted(group)) for group in groups)


def compile_csf(
    spec: Workload,
    readout_axes: np.ndarray | Sequence[np.ndarray],
    **kwargs,
) -> CSFCompilation:
    parameters = build_parameter_plans(spec, readout_axes, **kwargs)
    return compile_parameter_plans(parameters)


def compile_parameter_plans(parameters: Sequence[ParameterPlan]) -> CSFCompilation:
    parameters = tuple(parameters)
    parameter_ids = [parameter.parameter for parameter in parameters]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("CSF parameter ids must be unique")
    graph, reasons = consumer_conflict_graph(parameters)
    groups = _color_graph(graph)
    skipped = tuple(parameter.parameter for parameter in parameters if not parameter.consumer)
    compilation = CSFCompilation(
        parameters=parameters,
        groups=groups,
        conflicts={parameter: frozenset(neighbors) for parameter, neighbors in graph.items()},
        reason_counts=dict(reasons),
        skipped_parameters=skipped,
    )
    if not validate_compilation(compilation):
        raise RuntimeError("CSF produced an invalid fusion plan")
    return compilation


def validate_compilation(compilation: CSFCompilation) -> bool:
    active = {parameter.parameter for parameter in compilation.parameters if parameter.consumer}
    seen: set[int] = set()
    for group in compilation.groups:
        members = set(group)
        for parameter in group:
            if parameter in seen or parameter not in active:
                return False
            if compilation.conflicts.get(parameter, frozenset()) & (members - {parameter}):
                return False
            seen.add(parameter)
    return seen == active


# Compatibility helpers for existing experiments. They implement the original
# influence-only grouping; new code should call compile_csf with real readouts.
def conflict_graph(supports: list[frozenset[int]]) -> dict[int, set[int]]:
    graph = {parameter: set() for parameter in range(len(supports))}
    for left, support in enumerate(supports):
        for right in range(left):
            if support & supports[right]:
                graph[left].add(right)
                graph[right].add(left)
    return graph


def greedy_groups(supports: list[frozenset[int]]) -> list[tuple[int, ...]]:
    return list(_color_graph(conflict_graph(supports)))


def validate_groups(groups: list[tuple[int, ...]], supports: list[frozenset[int]]) -> bool:
    seen: set[int] = set()
    for group in groups:
        used: set[int] = set()
        for parameter in group:
            if parameter in seen or not supports[parameter].isdisjoint(used):
                return False
            seen.add(parameter)
            used.update(supports[parameter])
    return seen == set(range(len(supports)))
