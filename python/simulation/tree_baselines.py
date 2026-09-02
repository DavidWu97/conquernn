"""Tree-based quantile-regression baselines."""

from __future__ import annotations

import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor


def _levels(quantiles) -> tuple[np.ndarray, bool]:
    scalar = np.isscalar(quantiles)
    values = np.asarray([quantiles] if scalar else quantiles, dtype=float)
    if values.ndim != 1 or np.any((values <= 0) | (values >= 1)):
        raise ValueError(
            "quantiles must be a scalar or one-dimensional values in (0, 1)"
        )
    return values, scalar


def _split(n: int, fraction: float, random_state: int, splits=None):
    if splits is not None:
        return tuple(np.asarray(index, dtype=int) for index in splits)
    if not 0 < fraction < 1:
        raise ValueError("val_pct must be in (0, 1)")
    indices = np.arange(n)
    np.random.default_rng(random_state).shuffle(indices)
    cutoff = min(max(int(np.round(n * (1 - fraction))), 1), n - 1)
    return indices[:cutoff], indices[cutoff:]


def _as_matrix(prediction, n_quantiles: int) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    elif prediction.ndim == 3 and prediction.shape[1] == 1:
        prediction = prediction[:, 0, :]
    if prediction.ndim != 2 or prediction.shape[1] != n_quantiles:
        raise ValueError(f"Unsupported prediction shape: {prediction.shape}")
    return prediction


def _pinball(target, prediction, quantiles) -> float:
    target = np.asarray(target).reshape(-1, 1)
    prediction = _as_matrix(prediction, len(quantiles))
    residual = target - prediction
    levels = np.asarray(quantiles)[None, :]
    return float(np.mean(np.maximum(levels * residual, (levels - 1) * residual)))


class _StandardizedRegressor:
    def __init__(self, quantiles, random_state=0, val_pct=0.1, standardize=True):
        self.quantiles, self.scalar_quantile = _levels(quantiles)
        self.random_state = random_state
        self.val_pct = val_pct
        self.standardize = standardize
        self.x_mean = None

    def _fit_scaler(self, features, target):
        if self.standardize:
            self.x_mean = features.mean(axis=0, keepdims=True)
            self.x_std = features.std(axis=0, keepdims=True)
            self.x_std[self.x_std == 0] = 1
            self.y_mean = float(target.mean())
            self.y_std = float(target.std()) or 1.0
        else:
            self.x_mean = np.zeros((1, features.shape[1]))
            self.x_std = np.ones((1, features.shape[1]))
            self.y_mean, self.y_std = 0.0, 1.0

    def _transform_x(self, features):
        return (np.asarray(features, dtype=float) - self.x_mean) / self.x_std

    def _transform_y(self, target):
        return (np.asarray(target, dtype=float).reshape(-1) - self.y_mean) / self.y_std

    def _finish_prediction(self, prediction):
        prediction = _as_matrix(prediction, len(self.quantiles))
        prediction = prediction * self.y_std + self.y_mean
        return prediction[:, 0] if self.scalar_quantile else prediction

    def _prepare(self, features, target, splits, val_pct):
        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float).reshape(-1)
        if len(features) != len(target):
            raise ValueError(
                f"Incompatible shapes: {features.shape} and {target.shape}"
            )
        self._fit_scaler(features, target)
        transformed_x = self._transform_x(features)
        transformed_y = self._transform_y(target)
        indices = _split(
            len(target),
            self.val_pct if val_pct is None else val_pct,
            self.random_state,
            splits,
        )
        return transformed_x, transformed_y, indices


class QRF(_StandardizedRegressor):
    """Quantile regression forest based on ``quantile-forest``."""

    def __init__(
        self,
        quantiles=0.5,
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=1,
        max_depth=None,
        max_features=1.0,
        max_samples_leaf=1,
        bootstrap=True,
        n_jobs=-1,
        random_state=0,
        val_pct=0.1,
        standardize=True,
        enforce_monotone=False,
        loss_history="final",
        nepochs=None,
        **kwargs,
    ):
        super().__init__(quantiles, random_state, val_pct, standardize)
        default_quantiles = (
            float(self.quantiles[0])
            if self.scalar_quantile
            else self.quantiles.tolist()
        )
        self.model = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            default_quantiles=default_quantiles,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features=max_features,
            max_samples_leaf=max_samples_leaf,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )
        self.enforce_monotone = enforce_monotone
        self.loss_history = loss_history
        self.nepochs = nepochs
        self.label = "QRF"
        self.filename = "qrf"

    def _predict_scaled(self, features):
        levels = (
            float(self.quantiles[0])
            if self.scalar_quantile
            else self.quantiles.tolist()
        )
        prediction = _as_matrix(
            self.model.predict(features, quantiles=levels), len(self.quantiles)
        )
        if self.enforce_monotone and prediction.shape[1] > 1:
            prediction = np.sort(prediction, axis=1)
        return prediction

    def fit(self, X, y, splits=None, val_pct=None):
        features, target, (train, validation) = self._prepare(X, y, splits, val_pct)
        self.model.fit(features[train], target[train])
        train_loss = _pinball(
            target[train], self._predict_scaled(features[train]), self.quantiles
        )
        validation_loss = _pinball(
            target[validation],
            self._predict_scaled(features[validation]),
            self.quantiles,
        )
        train_losses = np.asarray([train_loss])
        validation_losses = np.asarray([validation_loss])
        if self.loss_history == "constant":
            length = self.nepochs if self.nepochs is not None else 1
            train_losses = np.full(length, train_loss)
            validation_losses = np.full(length, validation_loss)
        return train_losses, validation_losses

    def predict(self, X):
        if self.x_mean is None:
            raise RuntimeError("fit must be called before predict")
        return self._finish_prediction(self._predict_scaled(self._transform_x(X)))


class GBRTQuantile(_StandardizedRegressor):
    """Gradient-boosted trees fitted separately at each quantile level."""

    def __init__(
        self,
        quantiles=0.5,
        backend="gbrt",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=1.0,
        max_features=None,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        early_stopping=False,
        random_state=0,
        val_pct=0.1,
        standardize=True,
        enforce_monotone=False,
        **kwargs,
    ):
        super().__init__(quantiles, random_state, val_pct, standardize)
        if backend not in {"gbrt", "hist"}:
            raise ValueError("backend must be 'gbrt' or 'hist'")
        self.backend = backend
        self.parameters = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "subsample": subsample,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "l2_regularization": l2_regularization,
            "validation_fraction": validation_fraction,
            "n_iter_no_change": n_iter_no_change,
            "tol": tol,
            "early_stopping": early_stopping,
            "random_state": random_state,
        }
        self.model_kwargs = kwargs
        self.enforce_monotone = enforce_monotone
        self.models = None
        self.label = "GBRT-Quantile"
        self.filename = "gbrt_quantile"

    def _model(self, quantile):
        values = self.parameters
        if self.backend == "gbrt":
            return GradientBoostingRegressor(
                loss="quantile",
                alpha=float(quantile),
                n_estimators=values["n_estimators"],
                learning_rate=values["learning_rate"],
                max_depth=values["max_depth"],
                min_samples_split=values["min_samples_split"],
                min_samples_leaf=values["min_samples_leaf"],
                subsample=values["subsample"],
                max_features=values["max_features"],
                validation_fraction=values["validation_fraction"],
                n_iter_no_change=values["n_iter_no_change"],
                tol=values["tol"],
                random_state=values["random_state"],
                **self.model_kwargs,
            )
        parameters = dict(
            loss="quantile",
            quantile=float(quantile),
            max_iter=values["n_estimators"],
            learning_rate=values["learning_rate"],
            max_depth=values["max_depth"],
            max_leaf_nodes=values["max_leaf_nodes"],
            min_samples_leaf=values["min_samples_leaf"],
            l2_regularization=values["l2_regularization"],
            validation_fraction=values["validation_fraction"],
            early_stopping=values["early_stopping"],
            tol=values["tol"],
            random_state=values["random_state"],
        )
        if values["n_iter_no_change"] is not None:
            parameters["n_iter_no_change"] = values["n_iter_no_change"]
        if values["max_features"] is not None:
            parameters["max_features"] = values["max_features"]
        parameters.update(self.model_kwargs)
        return HistGradientBoostingRegressor(**parameters)

    def _predict_scaled(self, features):
        prediction = np.column_stack([model.predict(features) for model in self.models])
        if self.enforce_monotone and prediction.shape[1] > 1:
            prediction = np.sort(prediction, axis=1)
        return prediction

    def fit(self, X, y, splits=None, val_pct=None):
        features, target, (train, validation) = self._prepare(X, y, splits, val_pct)
        self.models = [self._model(level) for level in self.quantiles]
        for model in self.models:
            model.fit(features[train], target[train])
        train_loss = _pinball(
            target[train], self._predict_scaled(features[train]), self.quantiles
        )
        validation_loss = _pinball(
            target[validation],
            self._predict_scaled(features[validation]),
            self.quantiles,
        )
        return np.asarray([train_loss]), np.asarray([validation_loss])

    def predict(self, X):
        if self.x_mean is None or self.models is None:
            raise RuntimeError("fit must be called before predict")
        return self._finish_prediction(self._predict_scaled(self._transform_x(X)))
