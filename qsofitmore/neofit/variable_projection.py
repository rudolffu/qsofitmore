"""Bounded variable projection for separable nonlinear least squares."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares, lsq_linear


class VariableProjectionError(RuntimeError):
    """Raised when the bounded variable-projection solver cannot continue."""


DesignEvaluator = Callable[[np.ndarray, bool], Tuple[np.ndarray, Optional[Sequence[np.ndarray]]]]


@dataclass
class VariableProjectionState:
    """One cached nonlinear evaluation and its bounded linear solution."""

    nonlinear: np.ndarray
    design: np.ndarray
    derivatives: Optional[Tuple[np.ndarray, ...]]
    linear: np.ndarray
    linear_active_mask: np.ndarray
    residual: np.ndarray


@dataclass
class VariableProjectionResult:
    """Reduced optimizer result plus the final bounded linear solution."""

    nonlinear: np.ndarray
    linear: np.ndarray
    residual: np.ndarray
    reduced_jacobian: np.ndarray
    design: np.ndarray
    design_derivatives: Tuple[np.ndarray, ...]
    linear_active_mask: np.ndarray
    nonlinear_active_mask: np.ndarray
    success: bool
    status: int
    message: str
    nfev: int
    njev: int
    linear_solve_count: int


class _VariableProjectionProblem:
    def __init__(
        self,
        flux: np.ndarray,
        err: np.ndarray,
        linear_bounds: Tuple[np.ndarray, np.ndarray],
        evaluator: DesignEvaluator,
    ):
        self.flux = np.asarray(flux, dtype=float)
        self.err = np.asarray(err, dtype=float)
        self.weighted_flux = self.flux / self.err
        self.linear_lower = np.asarray(linear_bounds[0], dtype=float)
        self.linear_upper = np.asarray(linear_bounds[1], dtype=float)
        self.evaluator = evaluator
        self.linear_solve_count = 0
        self._state: Optional[VariableProjectionState] = None

    def state(self, nonlinear: np.ndarray, need_derivatives: bool) -> VariableProjectionState:
        nonlinear = np.asarray(nonlinear, dtype=float)
        if (
            self._state is not None
            and np.array_equal(nonlinear, self._state.nonlinear)
        ):
            if not need_derivatives or self._state.derivatives is not None:
                return self._state
            design, derivatives = self.evaluator(nonlinear, True)
            design = np.asarray(design, dtype=float)
            derivatives = tuple(np.asarray(item, dtype=float) for item in derivatives or ())
            if (
                design.shape != self._state.design.shape
                or not np.array_equal(design, self._state.design)
                or len(derivatives) != nonlinear.size
                or any(
                    item.shape != design.shape or not np.all(np.isfinite(item))
                    for item in derivatives
                )
            ):
                raise VariableProjectionError(
                    "Variable-projection derivative evaluation is inconsistent with its cached design."
                )
            self._state.derivatives = derivatives
            return self._state

        design, derivatives = self.evaluator(nonlinear, need_derivatives)
        design = np.asarray(design, dtype=float)
        if design.ndim != 2 or design.shape[0] != self.flux.size:
            raise VariableProjectionError("Variable-projection design matrix has an invalid shape.")
        if not np.all(np.isfinite(design)):
            raise VariableProjectionError("Variable-projection design matrix contains non-finite values.")
        if need_derivatives:
            if derivatives is None or len(derivatives) != nonlinear.size:
                raise VariableProjectionError("Variable-projection derivative count is invalid.")
            derivatives = tuple(np.asarray(item, dtype=float) for item in derivatives)
            if any(item.shape != design.shape or not np.all(np.isfinite(item)) for item in derivatives):
                raise VariableProjectionError("Variable-projection derivatives are invalid.")
        else:
            derivatives = None

        weighted_design = design / self.err[:, None]
        linear_result = lsq_linear(
            weighted_design,
            self.weighted_flux,
            bounds=(self.linear_lower, self.linear_upper),
            method="bvls",
        )
        self.linear_solve_count += 1
        if not linear_result.success or not np.all(np.isfinite(linear_result.x)):
            raise VariableProjectionError(
                f"Bounded linear solve failed: {getattr(linear_result, 'message', 'unknown error')}"
            )
        linear = np.asarray(linear_result.x, dtype=float)
        residual = self.weighted_flux - weighted_design @ linear
        if not np.all(np.isfinite(residual)):
            raise VariableProjectionError("Variable-projection residual contains non-finite values.")
        self._state = VariableProjectionState(
            nonlinear=nonlinear.copy(),
            design=design,
            derivatives=derivatives,
            linear=linear,
            linear_active_mask=np.asarray(linear_result.active_mask, dtype=int),
            residual=residual,
        )
        return self._state

    def residual(self, nonlinear: np.ndarray) -> np.ndarray:
        return self.state(nonlinear, need_derivatives=False).residual

    def jacobian(self, nonlinear: np.ndarray) -> np.ndarray:
        state = self.state(nonlinear, need_derivatives=True)
        weighted_design = state.design / self.err[:, None]
        free = state.linear_active_mask == 0
        free_design = weighted_design[:, free]
        free_information_inverse = (
            np.linalg.pinv(free_design.T @ free_design) if np.any(free) else None
        )
        jacobian = np.empty((self.flux.size, nonlinear.size), dtype=float)
        for index, derivative in enumerate(state.derivatives or ()):
            weighted_derivative = derivative / self.err[:, None]
            direct = weighted_derivative @ state.linear
            if np.any(free):
                rhs = weighted_derivative[:, free].T @ state.residual - free_design.T @ direct
                coefficient_derivative = free_information_inverse @ rhs
                jacobian[:, index] = -direct - free_design @ coefficient_derivative
            else:
                jacobian[:, index] = -direct
        if not np.all(np.isfinite(jacobian)):
            raise VariableProjectionError("Variable-projection Jacobian contains non-finite values.")
        return jacobian


def solve_variable_projection(
    flux: np.ndarray,
    err: np.ndarray,
    nonlinear_initial: np.ndarray,
    nonlinear_bounds: Tuple[np.ndarray, np.ndarray],
    linear_bounds: Tuple[np.ndarray, np.ndarray],
    evaluator: DesignEvaluator,
    *,
    jacobian_method: str = "semi_analytic",
    max_nfev: Optional[int] = None,
) -> VariableProjectionResult:
    """Solve a bounded separable nonlinear least-squares problem."""

    if jacobian_method not in ("semi_analytic", "2-point"):
        raise ValueError("jacobian_method must be 'semi_analytic' or '2-point'.")
    nonlinear_initial = np.asarray(nonlinear_initial, dtype=float)
    nonlinear_lower = np.asarray(nonlinear_bounds[0], dtype=float)
    nonlinear_upper = np.asarray(nonlinear_bounds[1], dtype=float)
    problem = _VariableProjectionProblem(flux, err, linear_bounds, evaluator)

    if nonlinear_initial.size:
        nonlinear_result = least_squares(
            problem.residual,
            nonlinear_initial,
            bounds=(nonlinear_lower, nonlinear_upper),
            jac=problem.jacobian if jacobian_method == "semi_analytic" else "2-point",
            max_nfev=max_nfev,
        )
        if not nonlinear_result.success:
            raise VariableProjectionError(str(nonlinear_result.message))
        final_state = problem.state(nonlinear_result.x, need_derivatives=True)
        reduced_jacobian = problem.jacobian(nonlinear_result.x)
        nonlinear_active_mask = np.asarray(nonlinear_result.active_mask, dtype=int)
        success = bool(nonlinear_result.success)
        status = int(nonlinear_result.status)
        message = str(nonlinear_result.message)
        nfev = int(nonlinear_result.nfev)
        njev = int(getattr(nonlinear_result, "njev", 0) or 0)
    else:
        final_state = problem.state(nonlinear_initial, need_derivatives=True)
        reduced_jacobian = np.empty((np.asarray(flux).size, 0), dtype=float)
        nonlinear_active_mask = np.empty(0, dtype=int)
        success = True
        status = 1
        message = "Bounded linear least-squares solution."
        nfev = 1
        njev = 0

    return VariableProjectionResult(
        nonlinear=final_state.nonlinear.copy(),
        linear=final_state.linear.copy(),
        residual=final_state.residual.copy(),
        reduced_jacobian=reduced_jacobian,
        design=final_state.design.copy(),
        design_derivatives=tuple(item.copy() for item in final_state.derivatives or ()),
        linear_active_mask=final_state.linear_active_mask.copy(),
        nonlinear_active_mask=nonlinear_active_mask,
        success=success,
        status=status,
        message=message,
        nfev=nfev,
        njev=njev,
        linear_solve_count=problem.linear_solve_count,
    )


def evaluate_profile_chi2(
    flux: np.ndarray,
    err: np.ndarray,
    nonlinear: np.ndarray,
    linear_bounds: Tuple[np.ndarray, np.ndarray],
    evaluator: DesignEvaluator,
) -> float:
    """Evaluate the bounded linear profile objective at fixed nonlinear values."""

    problem = _VariableProjectionProblem(flux, err, linear_bounds, evaluator)
    residual = problem.residual(np.asarray(nonlinear, dtype=float))
    return float(np.sum(residual**2))


def optimizer_result_adapter(
    *,
    full_x: np.ndarray,
    full_jacobian: np.ndarray,
    full_active_mask: np.ndarray,
    result: VariableProjectionResult,
):
    """Return the subset of ``OptimizeResult`` attributes used by neofit."""

    return SimpleNamespace(
        x=np.asarray(full_x, dtype=float),
        jac=np.asarray(full_jacobian, dtype=float),
        active_mask=np.asarray(full_active_mask, dtype=int),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        nfev=int(result.nfev),
        njev=int(result.njev),
        linear_solve_count=int(result.linear_solve_count),
    )
