'''
Runs the quantile regression benchmarks for different models and functions.
'''
import numpy as np
import multiprocessing as mp
import os

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


def run_experiment(params, device_info):

    import warnings
    warnings.filterwarnings("ignore")
    
    trial, scenario_idx, nidx, qidx, hidx, kidx, X_test, y_quantiles, \
            manual_grad, stop ,X_train, y_train, model = params
    
    device_type, device_id = device_info
    
    np.random.seed(42)
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(42)
    
    # Record running time
    start_time = time.time()
    
    model_id = f"{functions[0].__class__.__name__}_{model_shape}_N{sample_sizes[nidx]}_q{quantiles[qidx]}_{kernels[kidx]}_{'' if residual else 'no'}res"
    if kidx == 0:
        model_id += '_auto_nostop'
        train_loss, val_loss = model.fit(X_train, y_train)
    else:
        model_id += f"_h{bandwidths[hidx]}_{'manual' if manual_grad else 'auto'}_{'' if stop else 'no'}stop"
        train_loss, val_loss = model.fit(X_train, y_train, manual_grad=manual_grad, stop=stop)

    # Save the models for trial 1, sample size 10000 and quantile 0.5, baeline with 0.001 bandwidth & gaussian
    if trial == 0 and nidx == 2 and qidx == 2:
        if kidx == 0:
            torch.save(model, f'data/model/{model_id}')
        elif kidx == 1 and hidx == 4:
            torch.save(model, f'data/model/{model_id}')

    end_time = time.time()

    preds = model.predict(X_test)

    training_time = end_time - start_time
    
    mse_value = ((y_quantiles - preds)**2).mean(axis=0)[qidx]
    mae_value = np.abs(y_quantiles - preds).mean(axis=0)[qidx]
    
    # if trial % 5 == 0:
    #     print(f"Completed {trial+1} trials")

    return (trial, scenario_idx, hidx, kidx, nidx, qidx, 
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
    
    mse_results = np.full((N_trials, len(functions), len(bandwidths),len(kernels), len(sample_sizes), len(quantiles)), np.nan)
    mae_results = np.full((N_trials, len(functions), len(bandwidths),len(kernels), len(sample_sizes), len(quantiles)), np.nan)
    time_results = np.full((N_trials, len(functions), len(bandwidths),len(kernels), len(sample_sizes), len(quantiles)), np.nan)
    train_losses = np.full((N_trials, len(functions), len(bandwidths),len(kernels), len(sample_sizes), len(quantiles), 100), np.nan)
    val_losses = np.full((N_trials, len(functions), len(bandwidths), len(kernels), len(sample_sizes), len(quantiles), 100), np.nan)
   

    now_time = datetime.now().strftime("%Y%m%d%H%M")
    print(f"Time now: {now_time}")
    print(f"Shape of results: {mse_results.shape}")
    
    all_params = []
    
    for trial in range(N_trials):
        for scenario_idx, func in enumerate(functions):
           
            X_test = np.random.random(size=(N_test, func.n_in))
            y_test = func.sample(X_test)
            
            y_quantiles = np.array([func.quantile(X_test, q) for q in quantiles]).T

            for nidx, N_train in enumerate(sample_sizes):
                X_train = np.random.random(size=(N_train,func.n_in))
                y_train = func.sample(X_train)
                for qidx, q in enumerate(quantiles):
                    for kidx, kernel in enumerate(kernels):
                        if kidx == 0:
                            model = QuantileNetwork(quantiles=q,shape=model_shape,residual=residual)
                            params = (trial, scenario_idx, nidx, qidx, 0, kidx, X_test, y_quantiles, 
                                    manual_grad, stop, X_train, y_train, model)
                            all_params.append(params)
                        else:
                            for hidx, h in enumerate(bandwidths):
                                model = ConquerNetwork(quantiles=q,kernel=kernel,bandwidth=h,shape=model_shape,residual=residual)
                                params = (trial, scenario_idx, nidx, qidx, hidx, kidx, X_test, y_quantiles, 
                                    manual_grad, stop, X_train, y_train, model)
                                all_params.append(params)
    
    print(f"Total experiments to run: {len(all_params)}")
    
    device_list = [devices[i % len(devices)] for i in range(len(all_params))]
    
    num_processes = min(num_devices, len(all_params))
    print(f"Using {num_processes} processes")
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(run_experiment, [(params, device_list[i]) for i, params in enumerate(all_params)])
    
    for result in results:
        trial, scenario_idx, hidx, kidx, nidx, qidx, mse_value, mae_value, training_time, train_loss, val_loss = result
        
        if kidx == 0:
            mse_results[trial, scenario_idx, :, kidx, nidx, qidx] = np.tile(mse_value, len(bandwidths))
            mae_results[trial, scenario_idx, :, kidx, nidx, qidx] = np.tile(mae_value, len(bandwidths))
            time_results[trial, scenario_idx, :, kidx, nidx, qidx] = np.tile(training_time,len(bandwidths))
            train_losses[trial, scenario_idx, :, kidx, nidx, qidx, :] = np.tile(train_loss,(len(bandwidths),1))
            val_losses[trial, scenario_idx, :, kidx, nidx, qidx, :] = np.tile(val_loss,(len(bandwidths),1))
        else:
            mse_results[trial, scenario_idx, hidx, kidx, nidx, qidx] = mse_value
            mae_results[trial, scenario_idx, hidx, kidx, nidx, qidx] = mae_value
            time_results[trial, scenario_idx, hidx, kidx, nidx, qidx] = training_time
            train_losses[trial, scenario_idx, hidx, kidx, nidx, qidx, :] = train_loss
            val_losses[trial, scenario_idx, hidx, kidx, nidx, qidx, :] = val_loss

    
    model_id = f"{functions[0].__class__.__name__}_{model_shape}_trial{N_trials}_{'' if residual else 'no'}res_{'manual' if manual_grad else 'auto'}grad_{'' if stop else 'no'}stop"

    print(model_id)

    # print MSE and running time results table
    # scenario_idx = int(functions[0].__class__.__name__[-1])-1
    shape_idx = 0 if model_shape == (5,70) else 1
    for scenario_idx, func in enumerate(functions):
        scenario_name = int(functions[scenario_idx].__class__.__name__[-1])
        #print(f'Scenario {scenario_name}\n')
        for nidx, N_train in enumerate(sample_sizes):
            hidx = get_idx(scenario_name-1,shape_idx,nidx)
            print(f'\nSample size = {N_train}\n')

            print(f'\nMSE results:\n')
            np.set_printoptions(precision=4, suppress=True, floatmode='fixed')
            print(np.around(mse_results.mean(axis=0)[scenario_idx,hidx,:,nidx,:],4))

            print(f'\nMAE results:\n')
            np.set_printoptions(precision=4, suppress=True, floatmode='fixed')
            print(np.around(mae_results.mean(axis=0)[scenario_idx,hidx,:,nidx,:],4))


            print(f'\nRunning time:\n')
            np.set_printoptions(precision=2, suppress=True, floatmode='fixed')
            print(np.around(time_results.mean(axis=0)[scenario_idx,hidx,:,nidx,:],2))
            print('\n\n\n')

    result_path = f'data/model/{model_id}.npz'
    #result_path = f'data/{model_id}_{now_time}.npz'
    result_dict = {
        "mse_results": mse_results,
        "mae_results": mae_results,
        "time_results": time_results,
        "train_losses": train_losses,
        "val_losses": val_losses,
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