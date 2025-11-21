'''
Runs the quantile regression benchmarks for different models and functions.
'''
import numpy as np
import multiprocessing as mp
import os
from sklearn.model_selection import KFold

from scenario import Scenario1, Scenario2, Scenario3
from baseline import QuantileNetwork
from conquer_model import ConquerNetwork
from loss import PinballLoss, MultiQuantilePinballLoss
from utils import get_idx

import time
from datetime import datetime
import torch

# Settings
N_trials = 50
N_test = 10000
model_shape = (5,70)
sample_sizes = [1000,5000,10000]
quantiles = np.array([0.05,0.25,0.5,0.75,0.95])
functions = [Scenario2()]
bandwidths = [0.001,0.005,0.01,0.05,0.1]
kernels=['baseline','gaussian','uniform','epanechnikov']
manual_grad = True
stop = True
residual = False

'''K-fold cross-validation to select best bandwidth'''
def cross_validate_bandwidth(X_train, y_train, quantile, kernel, model_shape, residual, manual_grad, stop, k_folds=5):
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    bandwidth_losses = {h: [] for h in bandwidths}
    
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        for h in bandwidths:
            model = ConquerNetwork(
                quantiles=quantile, kernel=kernel, bandwidth=h,
                shape=model_shape, residual=residual
            )
            
            # training
            train_loss, val_loss = model.fit(
                X_train_fold, y_train_fold, 
                manual_grad=manual_grad, stop=stop
            )
            
            # calculate pinball loss on validation set
            preds = model.predict(X_val_fold)
            pinball_loss = np.mean(np.maximum(
                quantile * (y_val_fold - preds), 
                (quantile - 1) * (y_val_fold - preds)
            ))
            bandwidth_losses[h].append(pinball_loss)
    
    # average pinball loss for each h
    avg_losses = {h: np.mean(losses) for h, losses in bandwidth_losses.items()}
    
    # choose h with the smallest average loss
    best_bandwidth = min(avg_losses.items(), key=lambda x: x[1])[0]
    return best_bandwidth, avg_losses

def prepare_experiment_params(trial, scenario_idx, nidx, qidx, kidx, func, manual_grad, stop, residual):

    import warnings
    warnings.filterwarnings("ignore")
    
    np.random.seed(42 + trial)
    torch.manual_seed(42 + trial)
    
    X_test = np.random.random(size=(N_test, func.n_in))
    y_test = func.sample(X_test)
    y_quantiles = np.array([func.quantile(X_test, q) for q in quantiles]).T
    
    X_train = np.random.random(size=(sample_sizes[nidx], func.n_in))
    y_train = func.sample(X_train)
    
    if kidx == 0:
        # baseline
        model = QuantileNetwork(quantiles=quantiles[qidx], shape=model_shape, residual=residual)
        best_h = 0
    else:
        # conquer, find best h by K-fold CV
        best_h, avg_losses = cross_validate_bandwidth(
            X_train, y_train, quantiles[qidx], kernels[kidx], model_shape, 
            residual, manual_grad, stop
        )
        
        # conquer with selected h
        model = ConquerNetwork(
            quantiles=quantiles[qidx], kernel=kernels[kidx], bandwidth=best_h,
            shape=model_shape, residual=residual
        )
    
    return (trial, scenario_idx, nidx, qidx, best_h, kidx, X_test, y_quantiles, 
            manual_grad, stop, X_train, y_train, model)

def run_experiment(params, device_info):

    import warnings
    warnings.filterwarnings("ignore")
    
    trial, scenario_idx, nidx, qidx, best_h, kidx, X_test, y_quantiles, \
            manual_grad, stop, X_train, y_train, model = params
    
    device_type, device_id = device_info
    
    np.random.seed(42 + trial)
    torch.manual_seed(42 + trial)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(42 + trial)
    
    # Record running time
    start_time = time.time()
    
    model_id = f"{functions[0].__class__.__name__}_{model_shape}_N{sample_sizes[nidx]}_q{quantiles[qidx]}_{kernels[kidx]}_{'' if residual else 'no'}res"
    
    if kidx == 0:
        # baseline
        model_id += '_auto_nostop'
        train_loss, val_loss = model.fit(X_train, y_train)
    else:
        # conquer
        model_id += f"_besth{best_h}_{'manual' if manual_grad else 'auto'}_{'' if stop else 'no'}stop_CV"
        train_loss, val_loss = model.fit(X_train, y_train, manual_grad=manual_grad, stop=stop)

    # Save the models for trial 1, sample size 10000 and quantile 0.5
    if trial == 0 and nidx == 2 and qidx == 2:
        torch.save(model, f'data/model/{model_id}')
        #torch.save(model, f'test/{model_id}')

    end_time = time.time()

    preds = model.predict(X_test)

    training_time = end_time - start_time
    
    mse_value = ((y_quantiles - preds)**2).mean(axis=0)[qidx]
    mae_value = np.abs(y_quantiles - preds).mean(axis=0)[qidx]
    
    return (trial, scenario_idx, best_h, kidx, nidx, qidx, 
            mse_value, mae_value, training_time, train_loss, val_loss)

def run_benchmarks(demo=True):
    model_info = {'N_trial':N_trials, 'model_shape':model_shape,'sample_sizes':sample_sizes,
                  'quantiles':quantiles,'scenarios':[func.__class__.__name__ for func in functions],
                  'kernels':kernels, 'manual_grad':manual_grad,'stop':stop, 'residual':residual}

    print(f"N_trials = {N_trials}")
    print(f"Network shape = {model_shape}")
    print(f"Sample sizes = {sample_sizes}")
    print(f"Quantile levels = {quantiles}, single quantile")
    print(f"Scenarios = {[func.__class__.__name__ for func in functions]}")
    print(f"Kernels = {kernels}")
    if manual_grad:
        print('Using manual grad')
    else:
        print('Using auto grad')

    if stop:
        print('Using stop criterion')
    else:
        print("Using nostop criterion")

    if residual:
        print('Using residual-based structure')
    else:
        print("Not using residual-based structure")

    num_devices = mp.cpu_count()
    devices = [("cpu", 0) for _ in range(num_devices)]
    
    mse_results = np.full((N_trials, len(functions), len(kernels), len(sample_sizes), len(quantiles)), np.nan)
    mae_results = np.full((N_trials, len(functions), len(kernels), len(sample_sizes), len(quantiles)), np.nan)
    time_results = np.full((N_trials, len(functions), len(kernels), len(sample_sizes), len(quantiles)), np.nan)
    train_losses = np.full((N_trials, len(functions), len(kernels), len(sample_sizes), len(quantiles), 100), np.nan)
    val_losses = np.full((N_trials, len(functions), len(kernels), len(sample_sizes), len(quantiles), 100), np.nan)
    chosen_bandwidths = np.full((N_trials, len(functions), len(kernels), len(sample_sizes), len(quantiles)), np.nan)

    now_time = datetime.now().strftime("%Y%m%d%H%M")
    print(f"Time now: {now_time}")
    print(f"Shape of results: {mse_results.shape}")

    all_params = []
    for trial in range(N_trials):
        for scenario_idx, func in enumerate(functions):
            for nidx, N_train in enumerate(sample_sizes):
                for qidx, q in enumerate(quantiles):
                    for kidx, kernel in enumerate(kernels):
                        all_params.append((
                            trial, scenario_idx, nidx, qidx, kidx, func, 
                            manual_grad, stop, residual
                        ))
    
    print(f"Total experiments to run: {len(all_params)}")
    
    print("Preparing experiments with cross-validation...")
    num_processes = min(num_devices, len(all_params))
    print(f"Using {num_processes} processes for preparation")
    
    prepared_experiments = []
    with mp.Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.starmap(prepare_experiment_params, all_params)):
            prepared_experiments.append(result)
            if (i + 1) % 10 == 0:
                print(f"Prepared {i+1}/{len(all_params)} experiments")
    
    print("All experiments prepared, starting training...")
    
    device_list = [devices[i % len(devices)] for i in range(len(prepared_experiments))]
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(run_experiment, [(params, device_list[i]) for i, params in enumerate(prepared_experiments)])
    
    for result in results:
        trial, scenario_idx, best_h, kidx, nidx, qidx, mse_value, mae_value, training_time, train_loss, val_loss = result
        
        mse_results[trial, scenario_idx, kidx, nidx, qidx] = mse_value
        mae_results[trial, scenario_idx, kidx, nidx, qidx] = mae_value
        time_results[trial, scenario_idx, kidx, nidx, qidx] = training_time
        train_losses[trial, scenario_idx, kidx, nidx, qidx, :] = train_loss
        val_losses[trial, scenario_idx, kidx, nidx, qidx, :] = val_loss
        chosen_bandwidths[trial, scenario_idx, kidx, nidx, qidx] = best_h

    model_id = f"{functions[0].__class__.__name__}_{model_shape}_trial{N_trials}_{'' if residual else 'no'}res_{'manual' if manual_grad else 'auto'}grad_{'' if stop else 'no'}stop_CV"

    print(model_id)

    shape_idx = 0 if model_shape == (5,70) else 1
    for scenario_idx, func in enumerate(functions):
        scenario_name = int(functions[scenario_idx].__class__.__name__[-1])
        for nidx, N_train in enumerate(sample_sizes):
            #hidx = get_idx(scenario_name-1, shape_idx, nidx)
            print(f'\nSample size = {N_train}\n')

            print(f'\nMSE results:\n')
            np.set_printoptions(precision=4, suppress=True, floatmode='fixed')
            print(np.around(mse_results.mean(axis=0)[scenario_idx, :, nidx, :], 4))

            print(f'\nMAE results:\n')
            np.set_printoptions(precision=4, suppress=True, floatmode='fixed')
            print(np.around(mae_results.mean(axis=0)[scenario_idx, :, nidx, :], 4))

            print(f'\nRunning time:\n')
            np.set_printoptions(precision=2, suppress=True, floatmode='fixed')
            print(np.around(time_results.mean(axis=0)[scenario_idx, :, nidx, :], 2))
            
            print(f'\nChosen bandwidths for trial 1:\n')
            np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
            print(np.around(chosen_bandwidths[0, scenario_idx, :, nidx, :], 4))
            print('\n\n\n')

    result_path = f'data/model/{model_id}.npz'
    #result_path = f'test/{model_id}.npz'
    result_dict = {
        "mse_results": mse_results,
        "mae_results": mae_results,
        "time_results": time_results,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "chosen_bandwidths": chosen_bandwidths,
        "model_info": model_info
    }
    np.savez(result_path, **result_dict)

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    
    mp.set_start_method('spawn', force=True)
    
    np.set_printoptions(precision=4, suppress=True, floatmode='fixed')
    import warnings
    warnings.filterwarnings("ignore")
    
    run_benchmarks(demo=False)