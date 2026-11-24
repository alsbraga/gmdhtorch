"""Scikit-Learn compatible estimator wrappers (GMDHRegressor and GMDHClassifier) backed by GMDHModel and ModelIncrementer."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import r2_score
import torch
from torch import nn

from .network import (
    GMDHActivation,
    GMDHModel,
    GMDHNorm,
    GMDHOutput,
    combi_layers,
    default_layers,
)
from .selection import SelectionStrategy, resolve_selection_strategy
from .train import ModelIncrementer

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
except ImportError:  # pragma: no cover

    class BaseEstimator:  # type: ignore[no-redef]
        def get_params(self, deep: bool = True) -> dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.endswith("_")}

        def set_params(self, **params: Any) -> BaseEstimator:
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class RegressorMixin:  # type: ignore[no-redef]
        pass

    class ClassifierMixin:  # type: ignore[no-redef]
        pass


def _split_train_select(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    train_proportion: float = 2.0 / 3.0,
    shuffle_data: bool = False,
    rng: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples_count = x_arr.shape[0]
    n_train = int(math.floor(samples_count * float(train_proportion) + 0.5))
    all_idx = np.arange(samples_count)
    if shuffle_data:
        gen = np.random.default_rng(rng)
        all_idx = gen.permutation(samples_count)
    t_idx = all_idx[:n_train]
    s_idx = all_idx[n_train:]
    return x_arr[t_idx, :], y_arr[t_idx], x_arr[s_idx, :], y_arr[s_idx]


class GMDHRegressor(BaseEstimator, RegressorMixin):
    """Scikit-Learn compatible GMDH regressor wrapping ``GMDHModel`` and ``ModelIncrementer``."""

    def __init__(
        self,
        *,
        network_style: str = "default",
        forward_inputs: bool | None = None,
        min_inputs: int = 2,
        max_inputs: int = 2,
        max_degree: int = 2,
        interaction: int | None = None,
        min_layer_neurons: int = 1,
        max_layer_neurons: int = 30,
        min_layers: int = 0,
        max_layers: int = 10,
        select_method: SelectionStrategy | str = "select_median",
        performance_criterion: str = "MSE",
        min_grad: float = 1e-5,
        min_performance: float = 0.0,
        activation: nn.Module | type[nn.Module] | None = None,
        norm: GMDHNorm | bool | None = None,
        solver: str = "svd",
        rcond: float | None = 1e-12,
        alpha: float | str = 1e-6,
        equilibrate: bool = False,
        train_proportion: float = 2.0 / 3.0,
        shuffle_data: bool = False,
        random_state: int | None = None,
        logs: bool = False,
        device: str = "cpu",
        dtype: str | torch.dtype = "float64",
    ) -> None:
        self.network_style = network_style
        self.forward_inputs = forward_inputs
        self.min_inputs = min_inputs
        self.max_inputs = max_inputs
        self.max_degree = max_degree
        self.interaction = interaction
        self.min_layer_neurons = min_layer_neurons
        self.max_layer_neurons = max_layer_neurons
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.select_method = select_method
        self.performance_criterion = performance_criterion
        self.min_grad = min_grad
        self.min_performance = min_performance
        self.activation = activation
        self.norm = norm
        self.solver = solver
        self.rcond = rcond
        self.alpha = alpha
        self.equilibrate = equilibrate
        self.train_proportion = train_proportion
        self.shuffle_data = shuffle_data
        self.random_state = random_state
        self.logs = logs
        self.device = device
        self.dtype = dtype

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        select_X: np.ndarray | None = None,
        select_y: np.ndarray | None = None,
    ) -> GMDHRegressor:
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()

        if select_X is not None and select_y is not None:
            t_samples, t_targets = x_arr, y_arr
            s_samples = np.asarray(select_X, dtype=np.float64)
            s_targets = np.asarray(select_y, dtype=np.float64)
            if s_targets.ndim == 2 and s_targets.shape[1] == 1:
                s_targets = s_targets.ravel()
        else:
            t_samples, t_targets, s_samples, s_targets = _split_train_select(
                x_arr,
                y_arr,
                train_proportion=self.train_proportion,
                shuffle_data=self.shuffle_data,
                rng=self.random_state,
            )

        out_count = 1 if t_targets.ndim == 1 else int(t_targets.shape[1])
        style_low = str(self.network_style).strip().lower()
        is_combi = style_low in ("combi", "c-gmdh", "cgmdh")
        sel_strat = resolve_selection_strategy(self.select_method)

        if is_combi:
            eff_fwd = True if self.forward_inputs is None else bool(self.forward_inputs)
            layer_factory = combi_layers(
                max_neurons=self.max_layer_neurons,
                min_neurons=self.min_layer_neurons,
                max_inputs=self.max_inputs if self.max_inputs > 2 else 0,
                min_inputs=1,
                max_degree=self.max_degree,
                interaction=self.interaction,
                forward=eff_fwd,
            )
        else:
            eff_fwd = False if self.forward_inputs is None else bool(self.forward_inputs)
            layer_factory = default_layers(
                inputs=self.max_inputs,
                degree=self.max_degree,
                interaction=self.interaction,
                forward=eff_fwd,
                min_neurons=self.min_layer_neurons,
                max_neurons=self.max_layer_neurons,
            )

        stages: list[nn.Module] = []
        if self.norm:
            stages.append(
                GMDHNorm(dtype=self.dtype, device=self.device)
                if isinstance(self.norm, bool)
                else self.norm
            )
        if self.activation is not None and self.norm:
            stages.append(GMDHActivation(self.activation))
        stages.append(GMDHOutput(out_count, dtype=self.dtype, device=self.device))

        model = GMDHModel(*stages, dtype=self.dtype, device=self.device)
        inc = ModelIncrementer(
            model,
            layer=layer_factory,
            activation=self.activation,
            norm=self.norm,
            select=sel_strat,
            max_layers=self.max_layers,
            min_layers=self.min_layers,
            min_grad=self.min_grad,
            min_performance=self.min_performance,
            min_inputs=self.min_inputs if not is_combi else 1,
            criterion=self.performance_criterion,
            solver=self.solver,
            rcond=self.rcond,
            alpha=self.alpha,
            equilibrate=self.equilibrate,
            logs=self.logs,
            rng=self.random_state,
        )

        t_samples_t = torch.as_tensor(t_samples, dtype=model.dtype, device=model.device)
        t_targets_t = torch.as_tensor(t_targets, dtype=model.dtype, device=model.device)
        s_samples_t = torch.as_tensor(s_samples, dtype=model.dtype, device=model.device)
        s_targets_t = torch.as_tensor(s_targets, dtype=model.dtype, device=model.device)

        model.train()
        while inc.growing:
            inc.observe(t_targets_t, x=t_samples_t)
            inc.step([(s_samples_t, s_targets_t)])
        model.eval()

        self.model_: GMDHModel = model
        self.network_: GMDHModel = model
        self.selected_features_: np.ndarray = model.selected_inputs
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        out = self.model_.predict_numpy(x_arr)
        if out.ndim == 2 and out.shape[1] == 1:
            return out.ravel()
        return out

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(r2_score(np.asarray(y, dtype=np.float64).ravel(), y_pred.ravel()))


class GMDHClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-Learn compatible GMDH classifier using nearest-class-label rounding
    or One-vs-Rest GMDH ensembles for multi-class problems.
    """

    def __init__(
        self,
        *,
        mode: str = "regression_round",
        network_style: str = "default",
        forward_inputs: bool | None = None,
        max_layer_neurons: int = 30,
        max_layers: int = 10,
        select_method: SelectionStrategy | str = "select_median",
        solver: str = "svd",
        rcond: float | None = 1e-12,
        alpha: float | str = 1e-6,
        train_proportion: float = 2.0 / 3.0,
        shuffle_data: bool = False,
        random_state: int | None = None,
        logs: bool = False,
        device: str = "cpu",
        dtype: str | torch.dtype = "float64",
    ) -> None:
        self.mode = mode
        self.network_style = network_style
        self.forward_inputs = forward_inputs
        self.max_layer_neurons = max_layer_neurons
        self.max_layers = max_layers
        self.select_method = select_method
        self.solver = solver
        self.rcond = rcond
        self.alpha = alpha
        self.train_proportion = train_proportion
        self.shuffle_data = shuffle_data
        self.random_state = random_state
        self.logs = logs
        self.device = device
        self.dtype = dtype

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        select_X: np.ndarray | None = None,
        select_y: np.ndarray | None = None,
    ) -> GMDHClassifier:
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        self.classes_, y_idx = np.unique(y_arr, return_inverse=True)
        y_codes = (y_idx + 1).astype(np.float64)

        if self.mode == "ovr" and self.classes_.size > 2:
            self.estimators_: list[GMDHRegressor] = []
            for cls_code in range(1, self.classes_.size + 1):
                bin_y = (y_codes == cls_code).astype(np.float64)
                bin_sel_y = (
                    (np.searchsorted(self.classes_, np.asarray(select_y)) + 1 == cls_code).astype(np.float64)
                    if select_y is not None
                    else None
                )
                reg = GMDHRegressor(
                    network_style=self.network_style,
                    forward_inputs=self.forward_inputs,
                    max_layer_neurons=self.max_layer_neurons,
                    max_layers=self.max_layers,
                    select_method=self.select_method,
                    solver=self.solver,
                    rcond=self.rcond,
                    alpha=self.alpha,
                    train_proportion=self.train_proportion,
                    shuffle_data=self.shuffle_data,
                    random_state=self.random_state,
                    logs=self.logs,
                    device=self.device,
                    dtype=self.dtype,
                )
                reg.fit(x_arr, bin_y, select_X=select_X, select_y=bin_sel_y)
                self.estimators_.append(reg)
        else:
            sel_codes = (
                (np.searchsorted(self.classes_, np.asarray(select_y)) + 1).astype(np.float64)
                if select_y is not None
                else None
            )
            self.regressor_ = GMDHRegressor(
                network_style=self.network_style,
                forward_inputs=self.forward_inputs,
                max_layer_neurons=self.max_layer_neurons,
                max_layers=self.max_layers,
                select_method=self.select_method,
                solver=self.solver,
                rcond=self.rcond,
                alpha=self.alpha,
                train_proportion=self.train_proportion,
                shuffle_data=self.shuffle_data,
                random_state=self.random_state,
                logs=self.logs,
                device=self.device,
                dtype=self.dtype,
            )
            self.regressor_.fit(x_arr, y_codes, select_X=select_X, select_y=sel_codes)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        if hasattr(self, "estimators_"):
            scores = np.column_stack([est.predict(x_arr) for est in self.estimators_])
            best_idx = np.argmax(scores, axis=1)
            return self.classes_[best_idx]

        raw_out = self.regressor_.predict(x_arr)
        rounded = np.rint(raw_out).astype(np.int64) - 1
        clipped = np.clip(rounded, 0, self.classes_.size - 1)
        return self.classes_[clipped]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y)))
