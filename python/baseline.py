'''
A basic NN that optimizes the quantile loss for potentially-multiple quantiles.
'''
import numpy as np
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils import create_folds, batches
from torch_utils import clip_gradient, logsumexp

'''Residual block for residual-based network'''
class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.batchnorm = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        if input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.linear(x)
        out = self.dropout(out)
        out = self.batchnorm(out)
        out += identity
        out = self.relu(out)
        return out
    
'''Neural network to map from X to quantile(s) of y.'''
class QuantileNetworkModule(nn.Module):
    def __init__(self, X_means, X_stds, y_mean, y_std, n_out, shape, residual = False):

        # shape = (L,U), L = number of hidden layers, U = number of nodes in each layer

        super(QuantileNetworkModule, self).__init__()
        self.X_means = X_means
        self.X_stds = X_stds
        self.y_mean = y_mean
        self.y_std = y_std
        self.n_in = X_means.shape[1]
        self.n_out = n_out
        self.shape = shape
        self.residual = residual

        self.layers = []
        if self.residual:
            self.layers.append(ResidualBlock(self.n_in, shape[1]))
        else:
            self.layers.append(nn.Linear(self.n_in, shape[1]))
            self.layers.append(nn.Dropout(0.1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(shape[1]))
        for i in range(shape[0]-1):
            if self.residual:
                self.layers.append(ResidualBlock(shape[1], shape[1]))
            else:
                self.layers.append(nn.Linear(shape[1], shape[1]))
                self.layers.append(nn.Dropout(0.1))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm1d(shape[1]))
        self.layers.append(nn.Linear(shape[1], self.n_out if len(self.y_mean.shape) == 1 else self.n_out * self.y_mean.shape[1]))

        self.fc_in = nn.Sequential(*self.layers)
        self.softplus = nn.Softplus()

        
    def forward(self, x):
        fout = self.fc_in(x)

        # If we are dealing with multivariate responses, reshape to the (d x q) dimensions
        if len(self.y_mean.shape) != 1:
            fout = fout.reshape((-1, self.y_mean.shape[1], self.n_out))

        # If we only have 1 quantile, no need to do anything else
        if self.n_out == 1:
            return fout

        # Enforce monotonicity of the quantiles
        return torch.cat((fout[...,0:1], fout[...,0:1] + torch.cumsum(self.softplus(fout[...,1:]), dim=-1)), dim=-1)
        
    def predict(self, X):
        self.eval()
        self.zero_grad()
        tX = autograd.Variable(torch.FloatTensor((X - self.X_means) / self.X_stds), requires_grad=False)
        fout = self.forward(tX)
        return fout.data.numpy() * self.y_std[...,None] + self.y_mean[...,None]

class QuantileNetwork:
    def __init__(self, quantiles, loss='marginal', shape=(5,70), residual = False):
        self.quantiles = quantiles
        self.label = 'Quantile Network'
        self.filename = 'nn'
        self.lossfn = loss
        if self.lossfn != 'marginal':
            self.label += f' ({self.lossfn})'
        self.shape = shape
        self.residual = residual
        self.label += f' {self.shape}'
        self.label += ' res' if residual else ' nores'

    def fit(self, X, y):
        self.model, train_losses, val_losses = fit_quantiles(X, y, quantiles=self.quantiles, lossfn=self.lossfn, 
                                                             shape = self.shape, residual = self.residual)
        return train_losses, val_losses

    def predict(self, X):
        return self.model.predict(X)

def fit_quantiles(X, y, quantiles=0.5, lossfn = 'marginal',
                    shape = (5,70), nepochs=100, val_pct=0.1,
                    batch_size=None, target_batch_pct=0.01,
                    min_batch_size=20, max_batch_size=100,
                    verbose=False, lr=1e-1, weight_decay=0.0, patience=5,
                    init_model=None, splits=None, file_checkpoints=True,
                    clip_gradients=False, residual = False, **kwargs):
    if file_checkpoints:
        import uuid
        tmp_file = '/tmp/tmp_file_' + str(uuid.uuid4())

    if batch_size is None:
        batch_size = min(X.shape[0], max(min_batch_size, min(max_batch_size, int(np.round(X.shape[0]*target_batch_pct)))))
        if verbose:
            print('Auto batch size chosen to be {}'.format(batch_size))

    # Standardize the features and response (helps with gradient propagation)
    Xmean = X.mean(axis=0, keepdims=True)
    Xstd = X.std(axis=0, keepdims=True)
    Xstd[Xstd == 0] = 1 # Handle constant features
    ymean, ystd = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True)
    tX = autograd.Variable(torch.FloatTensor((X - Xmean) / Xstd), requires_grad=False)
    tY = autograd.Variable(torch.FloatTensor((y - ymean) / ystd), requires_grad=False)

    # Create train/validate splits
    if splits is None:
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices)*(1-val_pct)))
        train_indices = indices[:train_cutoff]
        validate_indices = indices[train_cutoff:]
    else:
        train_indices, validate_indices = splits

    if np.isscalar(quantiles):
        quantiles = np.array([quantiles])
    if lossfn == 'geometric':
        quantiles = 2*quantiles - 1
    tquantiles = autograd.Variable(torch.FloatTensor(quantiles), requires_grad=False)

    # Initialize the model
    model = QuantileNetworkModule(Xmean, Xstd, ymean, ystd, quantiles.shape[0], shape, residual) if init_model is None else init_model

    # Save the model to file
    if file_checkpoints:
        torch.save(model, tmp_file)
    else:
        import pickle
        model_str = pickle.dumps(model)

    # Setup the SGD method
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, nesterov=True, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Track progress
    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    num_bad_epochs = 0

    if verbose:
        print('ymax and min:', tY.max(), tY.min())

    # Univariate quantile loss
    def quantile_loss(yhat, tidx):
        z = tY[tidx,None] - yhat
        return torch.max(tquantiles[None]*z, (tquantiles[None] - 1)*z)

    # Marginal quantile loss for multivariate response
    def marginal_loss(yhat, tidx):
        z = tY[tidx,:,None] - yhat
        return torch.max(tquantiles[None,None]*z, (tquantiles[None,None] - 1)*z)

    # Geometric quantile loss -- uses a Euclidean unit ball definition of multivariate quantiles
    def geometric_loss(yhat, tidx):
        z = tY[tidx,:,None] - yhat
        return torch.norm(z, dim=1) + (z * tquantiles[None,None]).sum(dim=1)

    # Create the quantile loss function
    if len(tY.shape) == 1 or tY.shape[1] == 1:
        lossfn = quantile_loss
    elif lossfn == 'marginal':
        print('Using marginal loss')
        lossfn = marginal_loss
    elif lossfn == 'geometric':
        print('Using geometric loss')
        lossfn = geometric_loss
            

    # Train the model
    for epoch in range(nepochs):
        if verbose:
            print('\t\tEpoch {}'.format(epoch+1))
            sys.stdout.flush()

        # Track the loss curves
        train_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(train_indices, batch_size, shuffle=True)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tBatch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            # Set the model to training mode
            model.train()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the predicted quantiles
            yhat = model(tX[tidx])
            
            # Loss for all quantiles
            loss = lossfn(yhat, tidx).mean()

            # Calculate gradients
            loss.backward()

            # Clip the gradients
            if clip_gradients:
                clip_gradient(model)

            # Apply the update
            # [p for p in model.parameters() if p.requires_grad]
            optimizer.step()

            # Track the loss
            train_loss += loss.data

            if np.isnan(loss.data.numpy()):
                import warnings
                warnings.warn('NaNs encountered in training model.')
                break

        validate_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(validate_indices, batch_size, shuffle=False)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tValidation Batch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            # Set the model to test mode
            model.eval()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the conditional mixture weights
            yhat = model(tX[tidx])

            # Track the loss
            validate_loss += lossfn(yhat, tidx).sum()

        train_losses[epoch] = train_loss.data.numpy() / float(len(train_indices))
        val_losses[epoch] = validate_loss.data.numpy() / float(len(validate_indices))

        # Adjust the learning rate down if the validation performance is bad
        if num_bad_epochs > patience:
            if verbose:
                print('Decreasing learning rate to {}'.format(lr*0.5))
            scheduler.step(val_losses[epoch])
            lr *= 0.5
            num_bad_epochs = 0

        # If the model blew up and gave us NaNs, adjust the learning rate down and restart
        if np.isnan(val_losses[epoch]):
            if verbose:
                print('Network went to NaN. Readjusting learning rate down by 50%')
            if file_checkpoints:
                os.remove(tmp_file)
            return fit_quantiles(X, y, quantiles=quantiles, lossfn=lossfn,
                    nepochs=nepochs, val_pct=val_pct,
                    batch_size=batch_size, target_batch_pct=target_batch_pct,
                    min_batch_size=min_batch_size, max_batch_size=max_batch_size,
                    verbose=verbose, lr=lr*0.5, weight_decay=weight_decay, patience=patience,
                    init_model=init_model, splits=splits, file_checkpoints=file_checkpoints,  **kwargs)

        # Check if we are currently have the best held-out log-likelihood
        if epoch == 0 or val_losses[epoch] <= best_loss:
            if verbose:
                print('\t\t\tSaving test set results.      <----- New high water mark on epoch {}'.format(epoch+1))
            # If so, use the current model on the test set
            best_loss = val_losses[epoch]
            if file_checkpoints:
                torch.save(model, tmp_file)
            else:
                import pickle
                model_str = pickle.dumps(model)
        else:
            num_bad_epochs += 1
        
        if verbose:
            print('Validation loss: {} Best: {}'.format(val_losses[epoch], best_loss))

    # Load the best model and clean up the checkpoints
    if file_checkpoints:
        model = torch.load(tmp_file,weights_only=False)
        os.remove(tmp_file)
    else:
        import pickle
        model = pickle.loads(model_str)

    # Return the conditional density model that marginalizes out the grid
    return model, train_losses, val_losses