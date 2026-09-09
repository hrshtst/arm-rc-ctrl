# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Diagnosis of a failed ridge-equivalence comparison (M3REP-003; repetition plan section 6, condition D6).

A comparison of the numerical validation contrasts two fits whose ridge
problems are equal in exact arithmetic but accumulated differently: the
candidate stacks ``K`` literal copies of the episode, the reference the episode
once at the scaled parameter. When their predictions differ beyond the
approved tolerance, this module separates two sources of the gap with an
extended-precision reference solution:

1. *accumulation*: the candidate's normal matrix ``A_c = X_c^T X_c + alpha_c I``
   and right-hand side ``B_c`` against ``K`` times the reference's
   (``||A_c - K A_r||_F / ||A_c||_F`` and the same for ``B``), zero in exact
   arithmetic;
2. *solution path*: each fit's weights against the extended-precision (80-bit
   ``longdouble``) Cholesky reference solution of its own normal equations as
   this module reconstructs them in NumPy, in coefficients and in probe
   predictions. ``rclib`` assembles its normal equations with Eigen, so this
   part holds the assembly differences between the two paths as well as the
   LDLT solve's roundoff; it is not an isolated measurement of the solve;
3. the observed prediction gap against the gap between the two reference
   solutions (accumulation alone) and the first-order sensitivity
   ``cond2(A_r) * eps64``.

The diagnosis is reported, never used to relax a tolerance; the failed
comparison stays in the validation evidence as it is. The diagnosed
comparison's counterpart in the other formulation and the other pair of the
same entry and count are diagnosed alongside for contrast.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.execution import collect_execution, require_canonical
from arm_rc_ctrl.experiments.repetition_fits import CachedFit, FitStore
from arm_rc_ctrl.experiments.repetition_numerics import (
    COMPARISON_PAIRS,
    Comparison,
    NumericalValidation,
    PanelContext,
    ProbeMatrix,
    build_probes,
    load_validation,
    predict_probes,
)
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec
from arm_rc_ctrl.provenance import (
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
)
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.training import harvest_episode
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage
from arm_rc_ctrl.validation import is_hex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.experiments.repetition_fits import FitInputs
    from arm_rc_ctrl.experiments.repetition_panel import PanelEntry

__all__ = [
    "DIAGNOSIS_SCHEMA_VERSION",
    "ComparisonDiagnosis",
    "Diagnosis",
    "RidgeProblem",
    "SolveError",
    "diagnose_comparison",
    "extended_solve",
    "load_diagnosis",
    "main",
    "render_diagnosis_markdown",
    "ridge_problem",
    "run_diagnosis",
]

DIAGNOSIS_SCHEMA_VERSION: Final = 1
_SHA256_HEX: Final = 64
_MODULE: Final = "arm_rc_ctrl.experiments.repetition_diagnosis"
EPS64: Final = float(np.finfo(np.float64).eps)


def _frobenius(array: NDArray[np.floating]) -> float:
    return float(np.sqrt(np.sum(array.astype(np.longdouble) ** 2)))


@dataclass(frozen=True)
class RidgeProblem:
    """The float64 normal equations of one fit, as its harvested loss rows define them (bias column last)."""

    a: NDArray[np.float64]
    b: NDArray[np.float64]
    alpha: float
    rows: int


def ridge_problem(cached: CachedFit) -> RidgeProblem:
    """``A = X^T X + alpha I`` and ``B = X^T Y`` over the fit's actual loss rows (harvested from a reset)."""
    harvested = [harvest_episode(cached.model, episode) for episode in cached.episodes]
    states = np.vstack([h.training_states for h in harvested])
    targets = np.vstack([h.training_targets for h in harvested])
    x = np.hstack([states, np.ones((states.shape[0], 1), dtype=np.float64)])
    alpha = cached.recipe.solver_alpha
    return RidgeProblem(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ targets, alpha, int(x.shape[0]))


def extended_solve(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.longdouble]:
    """Solve the symmetric positive definite system ``A W = B`` by Cholesky factorization in ``longdouble``.

    The float64 inputs are taken as given, so the result is a reference
    solution of the reconstructed problem (accurate to roughly ``cond2``
    times the 80-bit epsilon), not the exact solution of the fit's problem.
    """
    n = a.shape[0]
    if a.shape != (n, n) or b.shape[0] != n:
        msg = f"A must be square and B conformable, got {a.shape} and {b.shape}"
        raise ValueError(msg)
    lower = np.zeros((n, n), dtype=np.longdouble)
    matrix = a.astype(np.longdouble)
    for j in range(n):
        diagonal = matrix[j, j] - np.sum(lower[j, :j] ** 2)
        if diagonal <= 0:
            msg = f"A is not positive definite (pivot {j} = {float(diagonal)!r})"
            raise ValueError(msg)
        lower[j, j] = np.sqrt(diagonal)
        if j + 1 < n:
            lower[j + 1 :, j] = (matrix[j + 1 :, j] - lower[j + 1 :, :j] @ lower[j, :j]) / lower[j, j]
    rhs = b.astype(np.longdouble)
    y = np.zeros_like(rhs)
    for i in range(n):
        y[i] = (rhs[i] - lower[i, :i] @ y[:i]) / lower[i, i]
    w = np.zeros_like(rhs)
    for i in range(n - 1, -1, -1):
        w[i] = (y[i] - lower[i + 1 :, i] @ w[i + 1 :]) / lower[i, i]
    return w


@dataclass(frozen=True)
class SolveError:
    """One fit's float64 weights against the reference solution of its NumPy-reconstructed normal equations.

    The difference holds every step that separates the two paths: ``rclib``'s
    Eigen assembly of the normal equations, its LDLT solve, and the
    reconstruction's own float64 assembly. It is not an isolated measurement
    of the solve's roundoff (C11).
    """

    coefficient_fro_rel: float
    """``||W - W*||_F / ||W*||_F``."""
    prediction_max_abs: float
    """Largest ``|P W - P W*|`` over the probe matrix (rad)."""
    normal_residual_extended: float
    """``||A W* - B||_F / (||A||_F ||W*||_F + ||B||_F)`` of the reference solution (its own accuracy)."""


@dataclass(frozen=True)
class ComparisonDiagnosis:
    """Where the gap between a candidate and its reference comes from."""

    panel_label: str
    formulation: str
    count: int
    candidate: str
    reference: str
    candidate_identity: str
    reference_identity: str
    failed: bool
    """Whether the validation recorded this comparison as failed (the others are diagnosed for contrast)."""
    observed_max_abs: float
    """The validation's largest elementwise prediction difference (rad; increments for residual fits)."""
    observed_coefficient_fro_rel: float
    cond2_reference: float
    sensitivity: float
    """``cond2(A_r) * eps64``: first-order relative sensitivity of the solution to float64 perturbations."""
    accumulation_a_rel: float
    """``||A_c - K A_r||_F / ||A_c||_F``: the accumulation discrepancy of the normal matrices."""
    accumulation_b_rel: float
    accumulation_bound: float
    """``cond2(A_r) * (acc. A rel + acc. B rel)``: the first-order bound on the accumulation's coefficient effect."""
    accumulation_coefficient_fro_rel: float
    """``||W_c* - W_r*||_F / ||W_r*||_F``: the gap between the two extended-precision solutions."""
    accumulation_prediction_max_abs: float
    """Largest ``|P W_c* - P W_r*|``: the part of the observed gap the accumulation alone explains (rad)."""
    candidate_solve: SolveError
    reference_solve: SolveError
    output_scale: float
    """``||B_r||_F / ||A_r||_F``, the magnitude the errors scale with (absolute targets versus increments)."""

    def __post_init__(self) -> None:
        """Identities are digests and the pair is one of the plan's."""
        for name in ("candidate_identity", "reference_identity"):
            if not is_hex(getattr(self, name), _SHA256_HEX):
                msg = f"{name} must be 64 lowercase hex characters"
                raise ValueError(msg)
        if (self.candidate.split("/")[1], self.reference.split("/")[1]) not in COMPARISON_PAIRS:
            msg = f"{self.candidate} against {self.reference} is not a comparison of the plan"
            raise ValueError(msg)


def diagnose_comparison(
    comparison: Comparison, candidate: CachedFit, reference: CachedFit, probes: ProbeMatrix
) -> ComparisonDiagnosis:
    """Diagnose one recorded comparison from its two cached fits and the entry's probe matrix."""
    if (candidate.record.identity, reference.record.identity) != (
        comparison.candidate_identity,
        comparison.reference_identity,
    ):
        msg = "the cached fits are not the comparison's candidate and reference"
        raise ValueError(msg)
    k = comparison.count
    problem_c, problem_r = ridge_problem(candidate), ridge_problem(reference)
    w_c_star = extended_solve(problem_c.a, problem_c.b)
    w_r_star = extended_solve(problem_r.a, problem_r.b)
    p = probes.states.astype(np.longdouble)
    bias = np.ones((p.shape[0], 1), dtype=np.longdouble)
    design = np.hstack([p, bias])

    def predict(weights: NDArray[np.longdouble] | NDArray[np.float64]) -> NDArray[np.longdouble]:
        return design @ weights.astype(np.longdouble)

    def solve_error(cached: CachedFit, problem: RidgeProblem, star: NDArray[np.longdouble]) -> SolveError:
        residual = _frobenius(problem.a.astype(np.longdouble) @ star - problem.b.astype(np.longdouble))
        denominator = _frobenius(problem.a) * _frobenius(star) + _frobenius(problem.b)
        return SolveError(
            coefficient_fro_rel=_frobenius(cached.weights.astype(np.longdouble) - star) / _frobenius(star),
            prediction_max_abs=float(np.max(np.abs(predict(cached.weights) - predict(star)))),
            normal_residual_extended=0.0 if denominator == 0.0 else residual / denominator,
        )

    eigenvalues = np.linalg.eigvalsh(problem_r.a)
    cond2 = float("inf") if eigenvalues[0] <= 0 else float(eigenvalues[-1] / eigenvalues[0])
    accumulation_a_rel = _frobenius(problem_c.a - k * problem_r.a) / _frobenius(problem_c.a)
    accumulation_b_rel = _frobenius(problem_c.b - k * problem_r.b) / _frobenius(problem_c.b)
    return ComparisonDiagnosis(
        panel_label=comparison.panel_label,
        formulation=comparison.formulation,
        count=k,
        candidate=comparison.candidate,
        reference=comparison.reference,
        candidate_identity=comparison.candidate_identity,
        reference_identity=comparison.reference_identity,
        failed=not comparison.passed,
        observed_max_abs=max(d.max_abs for d in comparison.differences),
        observed_coefficient_fro_rel=comparison.coefficient_fro_rel,
        cond2_reference=cond2,
        sensitivity=cond2 * EPS64,
        accumulation_a_rel=accumulation_a_rel,
        accumulation_b_rel=accumulation_b_rel,
        accumulation_bound=cond2 * (accumulation_a_rel + accumulation_b_rel),
        accumulation_coefficient_fro_rel=_frobenius(w_c_star - w_r_star) / _frobenius(w_r_star),
        accumulation_prediction_max_abs=float(np.max(np.abs(predict(w_c_star) - predict(w_r_star)))),
        candidate_solve=solve_error(candidate, problem_c, w_c_star),
        reference_solve=solve_error(reference, problem_r, w_r_star),
        output_scale=_frobenius(problem_r.b) / _frobenius(problem_r.a),
    )


@dataclass(frozen=True)
class Diagnosis:
    """The committed diagnosis of every failed comparison of a validation, with its contrasts."""

    validation_file: str
    validation_sha256: str
    execution_identity: str
    n_failed: int
    diagnoses: tuple[ComparisonDiagnosis, ...]
    provenance: ProvenanceRecord
    schema_version: int = field(default=DIAGNOSIS_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """The failed count re-derives from the records and every failed comparison is diagnosed first."""
        if self.schema_version != DIAGNOSIS_SCHEMA_VERSION:
            msg = f"unsupported diagnosis schema_version {self.schema_version}"
            raise ValueError(msg)
        if self.n_failed != sum(1 for d in self.diagnoses if d.failed):
            msg = "n_failed contradicts the diagnoses"
            raise ValueError(msg)
        if not is_hex(self.validation_sha256, _SHA256_HEX) or not is_hex(self.execution_identity, _SHA256_HEX):
            msg = "validation_sha256 and execution_identity must be 64 lowercase hex characters"
            raise ValueError(msg)


def _contrasts(validation: NumericalValidation, failed: Comparison) -> list[Comparison]:
    """The failed comparison, its counterpart in the other formulation, and the other pair at the same count."""
    selected: list[Comparison] = [failed]
    for other in validation.comparisons:
        if other is failed or other.panel_label != failed.panel_label or other.count != failed.count:
            continue
        same_pair = (other.candidate.split("/")[1], other.reference.split("/")[1]) == (
            failed.candidate.split("/")[1],
            failed.reference.split("/")[1],
        )
        if same_pair or other.formulation == failed.formulation:
            selected.append(other)
    return selected


def run_diagnosis(
    validation: NumericalValidation,
    entries: Sequence[PanelEntry],
    inputs: FitInputs,
    *,
    store: FitStore,
    validation_file: str,
    validation_sha256: str,
    provenance: ProvenanceRecord,
) -> Diagnosis:
    """Diagnose every failed comparison of ``validation`` (with contrasts) from the cached fits."""
    if inputs.execution_identity != validation.execution.identity:
        msg = "the diagnosis must run in the validation's execution environment (C10)"
        raise ValueError(msg)
    by_label = {entry.label: entry for entry in entries}
    selected: list[Comparison] = []
    for comparison in validation.comparisons:
        if not comparison.passed:
            selected.extend(c for c in _contrasts(validation, comparison) if c not in selected)
    diagnoses: list[ComparisonDiagnosis] = []
    probes: dict[str, ProbeMatrix] = {}
    for comparison in selected:
        entry = by_label[comparison.panel_label]
        arm_c = ArmSpec(comparison.formulation, comparison.candidate.split("/")[1], comparison.count - 1)
        reference_arm = comparison.reference.split("/")[1]
        arm_r = ArmSpec(comparison.formulation, reference_arm, None if reference_arm == "S" else comparison.count - 1)
        candidate = store.fit_or_load(entry, arm_c, inputs)
        reference = store.fit_or_load(entry, arm_r, inputs)
        if entry.label not in probes:
            single = store.fit_or_load(entry, ArmSpec("absolute", "S"), inputs)
            matrix = build_probes(
                single.model,
                single.recipe.encoder(),
                inputs.samples,
                scenario=inputs.scenario,
                warmup_s=entry.warmup_s,
                period_s=inputs.preprocessing.resample_period_s,
                derivative_method=inputs.preprocessing.derivative_method,
                task_code=inputs.samples.task_code,
                bank_count=validation.probe_bank_count,
            )
            recorded = validation.probes[entry.label]
            if [b.states_sha256 for b in matrix.banks] != [b.states_sha256 for b in recorded]:
                msg = f"{entry.label}: the rebuilt probe matrix differs from the validation's"
                raise ValueError(msg)
            probes[entry.label] = matrix
        diagnoses.append(diagnose_comparison(comparison, candidate, reference, probes[entry.label]))
        _check_observed(comparison, candidate, reference, probes[entry.label])
    return Diagnosis(
        validation_file=validation_file,
        validation_sha256=validation_sha256,
        execution_identity=validation.execution.identity,
        n_failed=sum(1 for d in diagnoses if d.failed),
        diagnoses=tuple(diagnoses),
        provenance=provenance,
    )


def _check_observed(comparison: Comparison, candidate: CachedFit, reference: CachedFit, probes: ProbeMatrix) -> None:
    """The cached fits reproduce the validation's recorded prediction gap exactly (same environment)."""
    gap = float(np.max(np.abs(predict_probes(candidate.model, probes) - predict_probes(reference.model, probes))))
    recorded = next(d.max_abs for d in comparison.differences if d.quantity in ("prediction", "increment"))
    if gap != recorded:
        msg = (
            f"{comparison.panel_label}/{comparison.candidate}: recomputed gap {gap!r} differs from the recorded "
            f"{recorded!r}"
        )
        raise ValueError(msg)


def diagnosis_to_json(diagnosis: Diagnosis) -> str:
    """Canonical JSON."""
    return canonical_json(to_mapping(diagnosis))


def load_diagnosis(path: Path) -> Diagnosis:
    """Strictly rebuild a diagnosis from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), Diagnosis)


def _e(value: float) -> str:
    return f"{value:.3e}"


def render_diagnosis_markdown(diagnosis: Diagnosis) -> str:
    """The Markdown rendering: one section per diagnosed comparison, failures first."""
    lines = [
        "# Task 1-a repetition numerical validation: diagnosis of failed comparisons (v1)",
        "",
        (
            f"Validation `{diagnosis.validation_file}` (sha256 `{diagnosis.validation_sha256[:12]}`), execution "
            f"identity `{diagnosis.execution_identity[:12]}`, project commit "
            f"`{diagnosis.provenance.project_commit[:12]}`{' (dirty)' if diagnosis.provenance.project_dirty else ''}: "
            f"{diagnosis.n_failed} failed comparison(s) diagnosed, each with its contrasts (the same pair in the "
            "other formulation and the other pair at the same count)."
        ),
        "",
        (
            "For each comparison the float64 normal equations of both fits are reconstructed in NumPy from "
            "their harvested loss rows and solved in 80-bit extended precision, giving a reference solution "
            "`W*` (not an exact solution). The observed gap decomposes into the accumulation discrepancy "
            "between the candidate's stacked problem and `K` times the reference's (exact-arithmetic zero) and "
            "each fit's difference from its reference solution; that difference holds rclib's Eigen assembly of "
            "the normal equations as well as its LDLT solve, so it is not an isolated measurement of the solve's "
            "roundoff. `cond2 * eps64` is the first-order sensitivity of the solution to float64 perturbations. "
            "Nothing here changes a tolerance or a decision of the validation."
        ),
        "",
        (
            "| entry | formulation | K | candidate | reference | failed | observed gap | coef rel | cond2 "
            "| cond2 eps | acc. A rel | acc. B rel | acc. bound | acc. coef rel | acc. gap | cand. solve rel "
            "| cand. solve gap | ref. solve rel | ref. solve gap | output scale |"
        ),
        (
            "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
            "| ---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    lines.extend(
        f"| {d.panel_label} | {d.formulation} | {d.count} | {d.candidate.split('/', 1)[1]} "
        f"| {d.reference.split('/', 1)[1]} | {'**yes**' if d.failed else 'no'} | {_e(d.observed_max_abs)} "
        f"| {_e(d.observed_coefficient_fro_rel)} | {_e(d.cond2_reference)} | {_e(d.sensitivity)} "
        f"| {_e(d.accumulation_a_rel)} | {_e(d.accumulation_b_rel)} | {_e(d.accumulation_bound)} "
        f"| {_e(d.accumulation_coefficient_fro_rel)} "
        f"| {_e(d.accumulation_prediction_max_abs)} | {_e(d.candidate_solve.coefficient_fro_rel)} "
        f"| {_e(d.candidate_solve.prediction_max_abs)} | {_e(d.reference_solve.coefficient_fro_rel)} "
        f"| {_e(d.reference_solve.prediction_max_abs)} | {_e(d.output_scale)} |"
        for d in diagnosis.diagnoses
    )
    lines += [
        "",
        (
            "Columns: *observed gap* is the validation's largest prediction (or increment) difference in rad; "
            "*coef rel* its Frobenius-relative coefficient difference; *acc.* the accumulation part (normal "
            "matrix, right-hand side, the first-order bound cond2 (A rel + B rel), extended-precision solutions, "
            "and their prediction gap); *solve rel* / "
            "*solve gap* each fit's float64 weights against the reference solution of its reconstructed normal "
            "equations (assembly differences included); *output scale* is `||B_r||_F / ||A_r||_F`."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point (launch pinned through ``arm_rc_ctrl.execution run``)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Diagnose the failed comparisons of a numerical validation.")
    parser.add_argument("--validation", type=str, required=True, help="numerical validation JSON")
    parser.add_argument("--manifest", type=str, required=True, help="frozen panel manifest JSON")
    parser.add_argument("--output", type=str, required=True, help="diagnosis JSON to write (must not exist)")
    parser.add_argument("--markdown", type=str, required=True, help="diagnosis Markdown to write (must not exist)")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    require_canonical()
    ensure_single_thread()
    for target in (args.output, args.markdown):
        if Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    root = repository_root()
    store = open_storage()
    validation_file = Path(cast("str", args.validation))
    validation = load_validation(validation_file)
    for name in ("numpy", "rclib"):
        __import__(name)
    execution = collect_execution(command=command_line(_MODULE, argv), role="main", now=datetime.now(tz=UTC))
    execution.check_canonical()
    context = PanelContext.load(Path(cast("str", args.manifest)), store=store, root=root, execution=execution)
    resolved = {
        "validation": sha256_file(validation_file),
        "execution_identity": execution.identity,
        "command": command_line(_MODULE, argv),
    }
    provenance = collect_provenance(
        resolved, seeds={}, artifacts=[context.payload], exploratory=bool(args.exploratory), now=datetime.now(tz=UTC)
    )
    require_clean_for_confirmatory(provenance)
    diagnosis = run_diagnosis(
        validation,
        context.manifest.entries,
        context.inputs,
        store=FitStore(store),
        validation_file=(
            validation_file.relative_to(root).as_posix()
            if validation_file.is_relative_to(root)
            else str(validation_file)
        ),
        validation_sha256=sha256_file(validation_file),
        provenance=provenance,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(diagnosis_to_json(diagnosis) + "\n", encoding="utf-8")
    Path(args.markdown).write_text(render_diagnosis_markdown(diagnosis), encoding="utf-8")
    print(json.dumps({"n_failed": diagnosis.n_failed, "diagnosed": len(diagnosis.diagnoses)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
