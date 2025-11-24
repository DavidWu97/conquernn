import numpy as np
import pandas as pd
import torch
import os
import re
from sklearn.model_selection import train_test_split, KFold
from baseline import QuantileNetwork
from conquer_model import ConquerNetwork
from loss import MultiQuantilePinballLoss, PinballLoss
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


np.random.seed(42)
torch.manual_seed(42)

file_path = "data/bmi" 
quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
kernels = ['gaussian', 'uniform', 'epanechnikov']
shape = (10,50)
bandwidths = [0.001,0.005,0.01,0.05,0.1]

def load_and_preprocess_data(file_path):

    df = pd.read_csv(f'{file_path}/health_lifestyle_dataset.csv')
    
    # Consider samples without famil history
    # Set X, y
    # Group the data by gender
    df = df[df['family_history'] == 0]
    features = ['age', 'daily_steps', 'sleep_hours', 'water_intake_l', 'calories_consumed']
    target = 'bmi'
    male_df = df[df['gender'] == 'Male']
    female_df = df[df['gender'] == 'Female']
    
    male_X = male_df[features].values
    male_y = male_df[target].values
    
    female_X = female_df[features].values
    female_y = female_df[target].values
    print(f"Sample shape for male: {male_X.shape}")
    print(f"Sample shape for female: {female_X.shape}")
    
    return male_X, male_y, female_X, female_y

# Pinball loss for single quantile level
def evaluate_pinball_loss_single(model, X_test, y_test, quantile):

    predictions = model.predict(X_test)
    
    if len(predictions.shape) == 3:
        predictions = predictions.squeeze(1)
    
    loss_fn = PinballLoss(quantile=quantile)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    
    if len(y_test_tensor.shape) == 1:
        y_test_tensor = y_test_tensor.unsqueeze(-1)
    
    loss = loss_fn(predictions_tensor, y_test_tensor)
    
    return loss.item()

# Pinball loss for multiple quantile levels
def evaluate_pinball_loss_multi(model, X_test, y_test, quantiles):

    predictions = model.predict(X_test)
    
    if len(predictions.shape) == 3:
        predictions = predictions.squeeze(1)
    
    losses = []
    for i, q in enumerate(quantiles):
        loss_fn = PinballLoss(quantile=q)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        pred_tensor = torch.tensor(predictions[:, i], dtype=torch.float32)

        if len(y_test_tensor.shape) == 1:
            y_test_tensor = y_test_tensor.unsqueeze(-1)
        
        loss = loss_fn(pred_tensor, y_test_tensor)
        losses.append(loss.item())
    
    return losses

def cross_validate_conquer(X_train, y_train, quantiles, kernel, k_folds=5, bandwidth_candidates=None, estimate='single'):

    if bandwidth_candidates is None:
        bandwidth_candidates = bandwidths
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    bandwidth_scores = {bw: [] for bw in bandwidth_candidates}
    
    print(f"Start cross-validation for selecting bandwidth (kernel = {kernel}, estimate = {estimate})...")
    
    for bandwidth in bandwidth_candidates:
        print(f"Evaluate bandwidth = {bandwidth}")
        
        fold_losses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"  Fold {fold+1}/{k_folds}")
            
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            conquer_model = ConquerNetwork(
                quantiles=quantiles,
                kernel=kernel,
                bandwidth=bandwidth,
                shape=shape,
                residual=False
            )
            
            conquer_model.fit(X_fold_train, y_fold_train)

            if estimate == 'single':
                val_loss = evaluate_pinball_loss_single(conquer_model, X_fold_val, y_fold_val, quantiles)
            else:
                val_losses = evaluate_pinball_loss_multi(conquer_model, X_fold_val, y_fold_val, quantiles)
                val_loss = np.mean(val_losses)
            
            fold_losses.append(val_loss)
        
        avg_loss = np.mean(fold_losses)
        bandwidth_scores[bandwidth] = avg_loss
        print(f"  bandwidth={bandwidth}, Mean Validation Loss: {avg_loss:.4f}")
    
    # Select bandwidth with smallest mean validation loss
    best_bandwidth = min(bandwidth_scores, key=bandwidth_scores.get)
    best_score = bandwidth_scores[best_bandwidth]
    
    print(f"Bset bandwidth: {best_bandwidth}, Mean Validation Loss: {best_score:.4f}")
    
    return best_bandwidth, bandwidth_scores

# Match models that have been trained and saved
def find_matching_files(folder_path, s):
    pattern = re.compile(rf"^{re.escape(s)}_(.+)_besth([\d.]+)$")
    matches = []
    
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            match = pattern.match(filename)
            if match:
                middle_string = match.group(1)
                number = float(match.group(2))
                matches.append((middle_string, number))
    
    return matches

def train_and_evaluate_gender(X_train, X_test, y_train, y_test, gender, quantiles, estimate='single'):

    print(f"\nProcess {gender} samples...")
    
    results = defaultdict(list)
    
    if estimate == 'single':
        # Train the model for each single quantile level
        for i, q in enumerate(quantiles):
            model_id_baseline = f"{gender}_{shape}_{estimate}q{q}_baseline"
            model_id_conquer = f"{gender}_{shape}_{estimate}q{q}_conquer"
            matches = find_matching_files(f"{file_path}/model",model_id_conquer)
            # If model has been trained, load existing file
            if matches:
                print(f"Loading existing baseline model for single quantile {q}: {model_id_baseline}")
                baseline_model = torch.load(f"{file_path}/model/{model_id_baseline}",weights_only=False)
                baseline_loss = evaluate_pinball_loss_single(baseline_model, X_test, y_test, q)
                results['baseline'].append(baseline_loss)
                for kernel, best_bw in matches:
                    print(f"Loading existing conquer model for single quantile {q}: {model_id_conquer}_{kernel}_besth{best_bw}")
                    conquer_model = torch.load(f"{file_path}/model/{model_id_conquer}_{kernel}_besth{best_bw}",weights_only=False)
                    conquer_loss = evaluate_pinball_loss_single(conquer_model, X_test, y_test, q)
                    results[kernel].append(conquer_loss)
            else:
                print(f"\nTrain Model for Single Quantile Level {q}...")
                
                # 1. Baseline model
                print("Train Baseline Model..")
                baseline_model = QuantileNetwork(
                    quantiles=q,
                    loss='marginal',
                    shape=shape,
                    residual=False
                )
                baseline_model.fit(X_train, y_train)
                baseline_loss = evaluate_pinball_loss_single(baseline_model, X_test, y_test, q)
                results['baseline'].append(baseline_loss)
                torch.save(baseline_model,f'{file_path}/model/{model_id_baseline}')
            
                # 2. Conquer models
                for kernel in kernels:
                    print(f"Train Conquer Model with {kernel} Kernel...")
                    
                    # Select best bandwidth by CV
                    best_bw, _ = cross_validate_conquer(
                        X_train, y_train, q, kernel, 
                        k_folds=5, estimate=estimate
                    )
                    
                    # Train the model with the best bandwidth
                    conquer_model = ConquerNetwork(
                        quantiles=q,
                        kernel=kernel,
                        bandwidth=best_bw,
                        shape=shape,
                        residual=False
                    )
                    conquer_model.fit(X_train, y_train)
                    
                    conquer_loss = evaluate_pinball_loss_single(conquer_model, X_test, y_test, q)
                    results[kernel].append(conquer_loss)
                    torch.save(conquer_model,f'{file_path}/model/{model_id_conquer}_{kernel}_besth{best_bw}')
                
    else:  # estimate == 'multi'
        model_id_baseline = f"{gender}_{shape}_{estimate}_baseline"
        model_id_conquer = f"{gender}_{shape}_{estimate}_conquer"
        matches = find_matching_files(f"{file_path}/model",model_id_conquer)
        if matches:
            print(f"Loading existing baseline model for multi quantiles: {model_id_baseline}")
            baseline_model = torch.load(f"{file_path}/model/{model_id_baseline}",weights_only=False)
            baseline_losses = evaluate_pinball_loss_multi(baseline_model, X_test, y_test, quantiles)
            results['baseline'] = baseline_losses
            for kernel, best_bw in matches:
                print(f"Loading existing conquer model for multi quantiles: {model_id_conquer}_{kernel}_besth{best_bw}")
                conquer_model = torch.load(f"{file_path}/model/{model_id_conquer}_{kernel}_besth{best_bw}",weights_only=False)
                conquer_losses = evaluate_pinball_loss_multi(conquer_model, X_test, y_test, quantiles)
                results[kernel] = conquer_losses
        else:
            print("\nTrain Model for Multiple Quantile Levels...")
            
            # 1. Baseline model
            print("Train Baseline Model...")
            baseline_model = QuantileNetwork(
                quantiles=quantiles,
                loss='marginal',
                shape=shape,
                residual=False
            )
            baseline_model.fit(X_train, y_train)
            baseline_losses = evaluate_pinball_loss_multi(baseline_model, X_test, y_test, quantiles)
            results['baseline'] = baseline_losses
            torch.save(baseline_model,f'{file_path}/model/{model_id_baseline}')
            
            # 2. Conquer models
            for kernel in kernels:
                print(f"Train Conquer Model with {kernel} Kernel...")
                
                best_bw, _ = cross_validate_conquer(
                    X_train, y_train, quantiles, kernel, 
                    k_folds=5, estimate=estimate
                )

                conquer_model = ConquerNetwork(
                    quantiles=quantiles,
                    kernel=kernel,
                    bandwidth=best_bw,
                    shape=shape,
                    residual=False
                )
                conquer_model.fit(X_train, y_train)

                conquer_losses = evaluate_pinball_loss_multi(conquer_model, X_test, y_test, quantiles)
                results[kernel] = conquer_losses
                torch.save(conquer_model,f'{file_path}/model/{model_id_conquer}_{kernel}_besth{best_bw}')
    
    return results

# Output results table
def create_results_table(male_results, female_results, quantiles, estimate):

    methods = ['baseline', 'gaussian', 'uniform', 'epanechnikov']
    
    if estimate == 'single':
        data = []
        for method in methods:
            row = [method]
            row.extend(male_results[method])
            data.append(row)
        
        for method in methods:
            row = [method]
            row.extend(female_results[method])
            data.append(row)
        
        columns = ['Method'] + [f'q{q}' for q in quantiles]
        df = pd.DataFrame(data, columns=columns)
        
        df.insert(0, 'Gender', ['Male'] * len(methods) + ['Female'] * len(methods))
        
    else:  # estimate == 'multi'
        data = []
        for method in methods:
            row = [method]
            row.extend(male_results[method])
            data.append(row)
        
        for method in methods:
            row = [method]
            row.extend(female_results[method])
            data.append(row)
        
        columns = ['Method'] + [f'q{q}' for q in quantiles]
        df = pd.DataFrame(data, columns=columns)
        
        df.insert(0, 'Gender', ['Male'] * len(methods) + ['Female'] * len(methods))
    
    return df

def main():
    # 1. Load data
    print("Load Data...")
    male_X, male_y, female_X, female_y = load_and_preprocess_data(file_path)
    
    # 2. Set quantile levels
    #quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    
    # 3. Split training/test data
    print("Split Training and Test Sets...")
    male_X_train, male_X_test, male_y_train, male_y_test = train_test_split(
        male_X, male_y, test_size=0.2, random_state=42
    )
    
    female_X_train, female_X_test, female_y_train, female_y_test = train_test_split(
        female_X, female_y, test_size=0.2, random_state=42
    )
    
    print(f"Sample Sizes for Male: Training Set {len(male_X_train)}, Test Set {len(male_X_test)}")
    print(f"Sample Sizes for Female: Training Set {len(female_X_train)}, Test set {len(female_X_test)}")
    
    # 4. Run single/multiple quantile models
    all_results = {}
    
    for estimate in ['single', 'multi']:
        print(f"\n{'='*50}")
        print(f"Process {estimate} Mode")
        print(f"{'='*50}")
        
        male_results = train_and_evaluate_gender(
            male_X_train, male_X_test, male_y_train, male_y_test,
            "Male", quantiles, estimate
        )
        female_results = train_and_evaluate_gender(
            female_X_train, female_X_test, female_y_train, female_y_test,
            "Female", quantiles, estimate
        )
        
        results_table = create_results_table(male_results, female_results, quantiles, estimate)
        all_results[estimate] = results_table
        
        print(f"\nResults for {estimate.upper()} Mode:")
        print(results_table.to_string(index=False))
        
        # Save the table to CSV file
        results_table.to_csv(f'{file_path}/results_{estimate}.csv', index=False)
    
    # Visualization
    for estimate in ['single', 'multi']:
        df = all_results[estimate]
        
        plot_data = pd.melt(df, id_vars=['Gender', 'Method'], 
                           value_vars=[f'q{q}' for q in quantiles],
                           var_name='Quantile', value_name='Pinball Loss')
        
        plot_data['Quantile_Value'] = plot_data['Quantile'].str.replace('q', '').astype(float)
        
        plt.figure(figsize=(12, 8))
        
        # Subplots for Male/Female
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Male
        male_data = plot_data[plot_data['Gender'] == 'Male']
        for method in male_data['Method'].unique():
            method_data = male_data[male_data['Method'] == method]
            ax1.plot(method_data['Quantile_Value'], method_data['Pinball Loss'], 
                    marker='o', label=method, linewidth=2)
        
        ax1.set_title(f'Male - {estimate.upper()} Mode')
        ax1.set_xlabel('Quantile')
        ax1.set_ylabel('Pinball Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Female
        female_data = plot_data[plot_data['Gender'] == 'Female']
        for method in female_data['Method'].unique():
            method_data = female_data[female_data['Method'] == method]
            ax2.plot(method_data['Quantile_Value'], method_data['Pinball Loss'], 
                    marker='o', label=method, linewidth=2)
        
        ax2.set_title(f'Female - {estimate.upper()} Mode')
        ax2.set_xlabel('Quantile')
        ax2.set_ylabel('Pinball Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{file_path}/plots/results_{estimate}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 6. Summarize
    print("\n" + "="*80)
    print("Summary of Final Results")
    print("="*80)
    
    for estimate in ['single', 'multi']:
        print(f"\n{estimate.upper()} Mode:")
        print(all_results[estimate].to_string(index=False))
        print()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
    