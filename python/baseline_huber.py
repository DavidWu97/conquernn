"""Huberized quantile-loss neural network used in Appendix Table 18."""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from baseline import QuantileNetworkModule
from torch_utils import clip_gradient
from utils import batches


class HuberQuantileNetwork:
    def __init__(
        self,
        quantiles,
        loss="marginal",
        huber_delta=0.05,
        shape=(5, 70),
        residual=False,
    ):
        self.quantiles = quantiles
        self.lossfn = loss
        self.huber_delta = huber_delta
        self.shape = shape
        self.residual = residual
        self.label = f"Huber Quantile Network {shape} delta={huber_delta}"
        self.filename = "nn_huber"

    def fit(self, X, y, stop=False):
        result = fit_quantiles(
            X,
            y,
            quantiles=self.quantiles,
            lossfn=self.lossfn,
            huber_delta=self.huber_delta,
            shape=self.shape,
            stop=stop,
            residual=self.residual,
        )
        self.model, train_losses, val_losses, self.best_epoch, self.stop_epoch = result
        return train_losses, val_losses

    def predict(self, X):
        return self.model.predict(X)


QuantileNetwork = HuberQuantileNetwork


def fit_quantiles(
    X,
    y,
    quantiles=0.5,
    lossfn="marginal",
    huber_delta=0.05,
    shape=(5, 70),
    stop=False,
    nepochs=100,
    val_pct=0.1,
    batch_size=None,
    target_batch_pct=0.01,
    min_batch_size=20,
    max_batch_size=500,
    verbose=False,
    lr=1e-1,
    weight_decay=0.0,
    patience=5,
    init_model=None,
    splits=None,
    file_checkpoints=False,
    clip_gradients=False,
    residual=False,
    **kwargs,
):
    # The reported Huber baseline trains for all epochs and returns the best
    # validation checkpoint. ``stop`` is retained for API compatibility.
    del stop
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    if lossfn == "geometric":
        raise NotImplementedError("Geometric Huber loss is not implemented")

    checkpoint = None
    if file_checkpoints:
        import uuid

        checkpoint = "/tmp/huber_quantile_" + str(uuid.uuid4())

    if batch_size is None:
        batch_size = min(
            X.shape[0],
            max(
                min_batch_size,
                min(max_batch_size, int(np.round(X.shape[0] * target_batch_pct))),
            ),
        )

    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    tx = autograd.Variable(torch.FloatTensor((X - x_mean) / x_std), requires_grad=False)
    ty = autograd.Variable(torch.FloatTensor((y - y_mean) / y_std), requires_grad=False)

    if splits is None:
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        cutoff = int(np.round(len(indices) * (1 - val_pct)))
        train_indices, validation_indices = indices[:cutoff], indices[cutoff:]
    else:
        train_indices, validation_indices = splits

    quantiles_input = quantiles
    levels = np.asarray([quantiles] if np.isscalar(quantiles) else quantiles)
    tlevels = autograd.Variable(torch.FloatTensor(levels), requires_grad=False)
    model = (
        QuantileNetworkModule(
            x_mean, x_std, y_mean, y_std, len(levels), shape, residual
        )
        if init_model is None
        else init_model
    )

    if checkpoint:
        torch.save(model, checkpoint)
    else:
        model_state = pickle.dumps(model)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        nesterov=True,
        momentum=0.9,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    train_losses = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    best_loss = None
    best_epoch = None
    bad_epochs = 0

    def huber_loss(residuals, levels_for_loss):
        absolute = torch.abs(residuals)
        smooth_absolute = torch.where(
            absolute <= huber_delta,
            residuals.pow(2) / (2 * huber_delta),
            absolute - huber_delta / 2,
        )
        return (
            torch.abs(levels_for_loss - (residuals < 0).type_as(residuals))
            * smooth_absolute
        )

    def scalar_loss(prediction, index):
        residuals = ty[index, None] - prediction
        return huber_loss(residuals, tlevels[None])

    def marginal_loss(prediction, index):
        residuals = ty[index, :, None] - prediction
        return huber_loss(residuals, tlevels[None, None])

    objective = scalar_loss if len(ty.shape) == 1 or ty.shape[1] == 1 else marginal_loss

    for epoch in range(nepochs):
        if verbose:
            print(f"Epoch {epoch + 1}")
            sys.stdout.flush()

        train_loss = torch.Tensor([0])
        for batch in batches(train_indices, batch_size, shuffle=True):
            index = autograd.Variable(torch.LongTensor(batch), requires_grad=False)
            model.train()
            model.zero_grad()
            loss = objective(model(tx[index]), index).mean()
            loss.backward()
            if clip_gradients:
                clip_gradient(model)
            optimizer.step()
            train_loss += loss.data
            if np.isnan(loss.data.numpy()):
                break

        validation_loss = torch.Tensor([0])
        for batch in batches(validation_indices, batch_size, shuffle=False):
            index = autograd.Variable(torch.LongTensor(batch), requires_grad=False)
            model.eval()
            model.zero_grad()
            validation_loss += objective(model(tx[index]), index).sum()

        train_losses[epoch] = train_loss.data.numpy() / len(train_indices)
        validation_losses[epoch] = validation_loss.data.numpy() / len(
            validation_indices
        )

        if bad_epochs > patience:
            scheduler.step(validation_losses[epoch])
            lr *= 0.5
            bad_epochs = 0

        if np.isnan(validation_losses[epoch]):
            if checkpoint:
                os.remove(checkpoint)
            return fit_quantiles(
                X,
                y,
                quantiles=quantiles_input,
                lossfn=lossfn,
                huber_delta=huber_delta,
                shape=shape,
                nepochs=nepochs,
                val_pct=val_pct,
                batch_size=batch_size,
                target_batch_pct=target_batch_pct,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                verbose=verbose,
                lr=lr * 0.5,
                weight_decay=weight_decay,
                patience=patience,
                init_model=init_model,
                splits=splits,
                file_checkpoints=file_checkpoints,
                clip_gradients=clip_gradients,
                residual=residual,
                **kwargs,
            )

        if best_loss is None or validation_losses[epoch] <= best_loss:
            best_loss = validation_losses[epoch]
            best_epoch = epoch + 1
            if checkpoint:
                torch.save(model, checkpoint)
            else:
                model_state = pickle.dumps(model)
        else:
            bad_epochs += 1

    if checkpoint:
        model = torch.load(checkpoint, weights_only=False)
        os.remove(checkpoint)
    else:
        model = pickle.loads(model_state)
    return model, train_losses, validation_losses, best_epoch, nepochs
