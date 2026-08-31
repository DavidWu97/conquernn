"""
A basic NN that optimizes the quantile loss for potentially-multiple quantiles.
"""

import os
import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from baseline import QuantileNetworkModule
from loss import QuantileLoss
from utils import batches


class ConquerNetwork:
    """Convolution-smoothed quantile neural network."""

    def __init__(
        self,
        quantiles,
        kernel="gaussian",
        bandwidth=0.05,
        shape=(5, 70),
        residual=False,
    ):
        self.quantiles = quantiles
        self.label = "Conquer Network"
        self.filename = "nn"
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.shape = shape
        self.residual = residual
        self.label += f"_{self.shape}"
        self.label += f"_(h={self.bandwidth})"
        self.label += f"_(q={self.quantiles})"
        self.label += "_res" if residual else "_nores"

    def fit(self, X, y, manual_grad=False, stop=False):
        self.model, train_losses, val_losses, self.best_epoch, self.stop_epoch = (
            fit_quantiles(
                X,
                y,
                quantiles=self.quantiles,
                kernel=self.kernel,
                bandwidth=self.bandwidth,
                shape=self.shape,
                manual_grad=manual_grad,
                stop=stop,
                residual=self.residual,
            )
        )
        return train_losses, val_losses

    def predict(self, X):
        return self.model.predict(X)


def fit_quantiles(
    X,
    y,
    quantiles=0.5,
    kernel="gaussian",
    bandwidth=0.05,
    shape=(5, 70),
    manual_grad=False,
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

    file_path = f"data/LOG_{shape}_{kernel}_h{bandwidth}_q{quantiles}"

    if file_checkpoints:
        import uuid

        tmp_file = "/tmp/tmp_file_" + str(uuid.uuid4())

    if batch_size is None:
        batch_size = min(
            X.shape[0],
            max(
                min_batch_size,
                min(max_batch_size, int(np.round(X.shape[0] * target_batch_pct))),
            ),
        )
        if verbose:
            print("Auto batch size chosen to be {}".format(batch_size))

    Xmean = X.mean(axis=0, keepdims=True)
    Xstd = X.std(axis=0, keepdims=True)
    Xstd[Xstd == 0] = 1  # Handle constant features
    ymean, ystd = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True)
    tX = autograd.Variable(torch.FloatTensor((X - Xmean) / Xstd), requires_grad=False)
    tY = autograd.Variable(torch.FloatTensor((y - ymean) / ystd), requires_grad=False)

    if splits is None:
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices) * (1 - val_pct)))
        train_indices = indices[:train_cutoff]
        validate_indices = indices[train_cutoff:]
    else:
        train_indices, validate_indices = splits

    if np.isscalar(quantiles):
        quantiles = np.array([quantiles])
    tquantiles = autograd.Variable(torch.FloatTensor(quantiles), requires_grad=False)

    model = (
        QuantileNetworkModule(
            Xmean, Xstd, ymean, ystd, quantiles.shape[0], shape, residual
        )
        if init_model is None
        else init_model
    )

    if file_checkpoints:
        torch.save(model, tmp_file)
    else:
        import pickle

        model_str = pickle.dumps(model)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        nesterov=True,
        momentum=0.9,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    best_epoch = None
    stop_epoch = nepochs
    num_bad_epochs = 0

    if verbose:
        print("ymax and min:", tY.max(), tY.min())

    bias = torch.tensor(0)
    lossfn = None
    lossfunction = QuantileLoss(quantiles=tquantiles, bandwidth=bandwidth)
    if kernel == "gaussian":
        lossfn = lossfunction.gaussian
    elif kernel == "logistic":
        lossfn = lossfunction.logistic
    elif kernel == "uniform":
        lossfn = lossfunction.uniform
    elif kernel == "epanechnikov":
        lossfn = lossfunction.epanechnikov
    elif kernel == "triangular":
        lossfn = lossfunction.triangular
    else:
        print("Current kernel not implemented")
        return

    new_lr = lr
    for epoch in range(nepochs):
        if verbose:
            print("\t\tEpoch {}".format(epoch + 1))
            sys.stdout.flush()

        train_loss = torch.Tensor([0])
        model.train()
        for batch_idx, batch in enumerate(
            batches(train_indices, batch_size, shuffle=True)
        ):
            if verbose and (batch_idx % 100 == 0):
                print("\t\t\tBatch {}".format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            def closure():
                optimizer.zero_grad()
                yhat = model(tX[tidx])
                if manual_grad:
                    loss, grad_z = lossfn(
                        yhat, tY[tidx, None], requires_grad=manual_grad
                    )
                    grad_yhat = -grad_z / (len(tidx) * yhat.shape[1])
                    yhat.backward(grad_yhat)
                    return loss
                else:
                    loss = lossfn(yhat, tY[tidx, None], requires_grad=manual_grad)
                    loss.backward()
                    return loss

            train_loss += optimizer.step(closure) * len(tidx)

        validate_loss = torch.Tensor([0])
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                batches(validate_indices, batch_size, shuffle=False)
            ):
                if verbose and (batch_idx % 100 == 0):
                    print("\t\t\tValidation Batch {}".format(batch_idx))
                tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

                yhat = model(tX[tidx])

                validate_loss += lossfn(
                    yhat, tY[tidx, None], requires_grad=False
                ) * len(tidx)

        train_losses[epoch] = train_loss.data.cpu().numpy() / float(len(train_indices))
        val_losses[epoch] = validate_loss.data.cpu().numpy() / float(
            len(validate_indices)
        )

        if np.isnan(val_losses[epoch]):
            if verbose:
                print("Network went to NaN. Readjusting learning rate down by 50%")
            if file_checkpoints:
                os.remove(tmp_file)
            return fit_quantiles(
                X,
                y,
                quantiles=quantiles,
                lossfn=lossfn,
                kernel=kernel,
                bandwidth=bandwidth,
                shape=shape,
                manual_grad=manual_grad,
                stop=stop,
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
                **kwargs,
            )

        if epoch == 0 or val_losses[epoch] <= best_loss:
            num_bad_epochs = 0
            if verbose:
                print(
                    "\t\t\tSaving test set results.      <----- New high water mark on epoch {}".format(
                        epoch + 1
                    )
                )
            best_loss = val_losses[epoch]
            best_epoch = epoch + 1
            if file_checkpoints:
                torch.save(model, tmp_file)
            else:
                import pickle

                model_str = pickle.dumps(model)
        else:
            num_bad_epochs += 1

        if verbose:
            print("Validation loss: {} Best: {}".format(val_losses[epoch], best_loss))

        if stop:
            if new_lr < 1e-4 and num_bad_epochs >= patience:
                stop_epoch = epoch + 1
                break

        old_lr = new_lr
        scheduler.step(val_losses[epoch])
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr != old_lr:
            num_bad_epochs = 0

    if file_checkpoints:
        model = torch.load(tmp_file, weights_only=False)
        os.remove(tmp_file)
    else:
        import pickle

        model = pickle.loads(model_str)

    if stop:
        train_losses = train_losses[:stop_epoch]
        val_losses = val_losses[:stop_epoch]

    return model, train_losses, val_losses, best_epoch, stop_epoch
