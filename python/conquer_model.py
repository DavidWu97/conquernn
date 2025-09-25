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
from baseline import QuantileNetworkModule
from utils import create_folds, batches
from torch_utils import clip_gradient, logsumexp

#from loss import QuantileLoss
from loss import QuantileLoss


'''Choose kernel from ['gaussian','uniform','triangular','epanechnikov']'''
class ConquerNetwork:
    def __init__(self, quantiles, kernel='gaussian', bandwidth=0.05, shape=(5,70)):
        self.quantiles = quantiles
        self.label = 'Conquer Network'
        self.filename = 'nn'
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.shape = shape
        self.label += f'_{self.shape}'
        self.label += f'_(h={self.bandwidth})'
        self.label += f'_(q={self.quantiles})'

            
    def fit(self, X, y, manual_grad=False, stop = False):
        self.model, train_losses, val_losses = fit_quantiles(
            X, y, quantiles=self.quantiles, kernel=self.kernel, 
            bandwidth=self.bandwidth, shape=self.shape,
            manual_grad=manual_grad, stop = stop
            )
        return train_losses, val_losses

    def predict(self, X):
        return self.model.predict(X)

def fit_quantiles(X, y, quantiles=0.5, kernel='gaussian',bandwidth=0.05, 
                  shape=(5,70), manual_grad=False, stop = False,
                  nepochs=100, val_pct=0.1, batch_size=None, target_batch_pct=0.01,
                  min_batch_size=20, max_batch_size=100, verbose=False, lr=1e-1, 
                  weight_decay=0.0, patience=5, init_model=None, splits=None, 
                  file_checkpoints=True, clip_gradients=False, **kwargs):

    file_path = f"data/LOG_{shape}_{kernel}_h{bandwidth}_q{quantiles}"
    
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
    Xstd[Xstd == 0] = 1  # Handle constant features
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
    tquantiles = autograd.Variable(torch.FloatTensor(quantiles), requires_grad=False)

    # Initialize the model
    model = QuantileNetworkModule(Xmean, Xstd, ymean, ystd, quantiles.shape[0], shape = shape) if init_model is None else init_model

    # Save the model to file
    if file_checkpoints:
        torch.save(model, tmp_file)
    else:
        import pickle
        model_str = pickle.dumps(model)


    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, nesterov=True, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor = 0.5,patience = 5)

    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    num_bad_epochs = 0

    if verbose:
        print('ymax and min:', tY.max(), tY.min())

    # Create the quantile loss function
    bias = torch.tensor(0)
    lossfn = None
    lossfunction = QuantileLoss(quantiles=tquantiles, bandwidth=bandwidth)
    if kernel == 'gaussian':
        #from scipy.stats import norm
        #print('Using gaussian smooth loss')
        lossfn = lossfunction.gaussian 
        #biasx = torch.tensor(-bandwidth * norm.ppf(quantiles))
        #bias = lossfn(torch.tensor(0), biasx, requires_grad=False)
    elif kernel == 'logistic':
        #print('Using logistic smooth loss')
        lossfn = lossfunction.logistic
        #biasx = torch.tensor(-bandwidth * np.log((1+quantiles)/(1-quantiles)))
        #bias = lossfn(torch.tensor(0), biasx, requires_grad=False)
    elif kernel == 'uniform':
        #print('Using uniform smooth loss')
        lossfn = lossfunction.uniform
        #biasx = torch.tensor(-bandwidth * (2*quantiles-1))
        #bias = lossfn(torch.tensor(0), biasx, requires_grad=False)
    elif kernel == 'epanechnikov':
        #print('Using epanechnikov smooth loss')
        lossfn = lossfunction.epanechnikov
        # import sympy as sp
        # x = sp.Symbol('x')
        # f = (x/bandwidth)**3 - 3*(x/bandwidth) + 2-4*quantiles
        # solutions = sp.solve(f)
        # biasx = torch.tensor(float(sp.re(solutions[1][x])))
        # bias = lossfn(torch.tensor(0), biasx, requires_grad=False)
    elif kernel == 'triangular':
        #print('Using triangular smooth loss')
        lossfn = lossfunction.triangular
        #biasx = torch.tensor(bandwidth * (1 - np.sqrt(2*quantiles)) if quantiles < 0.5 else bandwidth * (np.sqrt(2-2*quantiles) - 1)).to(device)
        #bias = lossfn(torch.tensor(0), biasx, requires_grad=False)
    else:
        print('Current kernel not implemented')
        return

    new_lr = lr
    for epoch in range(nepochs):
        if verbose:
            print('\t\tEpoch {}'.format(epoch+1))
            sys.stdout.flush()

        # Track the loss curves
        train_loss = torch.Tensor([0])
        model.train()
        for batch_idx, batch in enumerate(batches(train_indices, batch_size, shuffle=True)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tBatch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            def closure():
                optimizer.zero_grad()
                yhat = model(tX[tidx])
                if manual_grad:
                    loss, grad_z = lossfn(yhat, tY[tidx,None], requires_grad=manual_grad)
                    grad_yhat = -grad_z / len(tidx)
                    yhat.backward(grad_yhat)
                    return loss
                else:
                    loss = lossfn(yhat, tY[tidx,None], requires_grad=manual_grad)
                    loss.backward()
                    return loss
        
            train_loss += optimizer.step(closure) * len(tidx)


        validate_loss = torch.Tensor([0])
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(batches(validate_indices, batch_size, shuffle=False)):
                if verbose and (batch_idx % 100 == 0):
                    print('\t\t\tValidation Batch {}'.format(batch_idx))
                tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

                # Run the model and get the conditional mixture weights
                yhat = model(tX[tidx])

                # Track the loss
                validate_loss += lossfn(yhat, tY[tidx,None], requires_grad=False) * len(tidx)

        train_losses[epoch] = train_loss.data.cpu().numpy() / float(len(train_indices))
        val_losses[epoch] = validate_loss.data.cpu().numpy() / float(len(validate_indices))


        # If the model blew up and gave us NaNs, adjust the learning rate down and restart
        if np.isnan(val_losses[epoch]):
            if verbose:
                print('Network went to NaN. Readjusting learning rate down by 50%')
            if file_checkpoints:
                os.remove(tmp_file)
            return fit_quantiles(X, y, quantiles=quantiles, lossfn=lossfn, kernel=kernel,
                    bandwidth=bandwidth, shape=shape, manual_grad=manual_grad, 
                    stop = stop, nepochs=nepochs, val_pct=val_pct, batch_size=batch_size, 
                    target_batch_pct=target_batch_pct, min_batch_size=min_batch_size, 
                    max_batch_size=max_batch_size, verbose=verbose, lr=lr*0.5, 
                    weight_decay=weight_decay, patience=patience, init_model=init_model, 
                    splits=splits, file_checkpoints=file_checkpoints, **kwargs)

        # Check if we are currently have the best held-out log-likelihood
        if epoch == 0 or val_losses[epoch] <= best_loss:
            num_bad_epochs = 0
            if verbose:
                print('\t\t\tSaving test set results.      <----- New high water mark on epoch {}'.format(epoch+1))
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

        # stop criterion
        if stop:
            if new_lr < 1e-4 and num_bad_epochs >= patience:
                break

        old_lr = new_lr
        scheduler.step(val_losses[epoch])
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != old_lr:
            num_bad_epochs = 0


    # Load the best model and clean up the checkpoints
    if file_checkpoints:
        model = torch.load(tmp_file,weights_only=False)
        os.remove(tmp_file)
    else:
        import pickle
        model = pickle.loads(model_str)


    # Return the conditional density model that marginalizes out the grid
    return model, train_losses, val_losses