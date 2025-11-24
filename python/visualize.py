import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from utils import filter_h

scenario_names = ['Scenario 1', 'Scenario 2', 'Scenario 3']
shape_names = ['Model A', 'Model B']
bandwidth_names =['0.001', '0.005', '0.01', '0.05', '0.1']
kernel_names = ['Baseline', 'Gaussian', 'Uniform', 'Epanechnikov']
sample_size_names = ['1000', '5000', '10000']
quantile_names = ['0.05', '0.25', '0.5', '0.75', '0.95']

# gererate tables of MSE or running time
def table_full(data,file_path):

    # data shape (trial,scenario,model shape,bandwidth,kernel,samplesize,quantile)

    averaged_data = np.mean(data, axis=0)
    all_tables = []
    
    for scenario_idx in range(3):
        for model_shape_idx in range(2):
            table_data = []
            current_data = averaged_data[scenario_idx, model_shape_idx]
            
            baseline_value = None
            for bw_idx, bandwidth in enumerate(bandwidth_names):
                for kernel_idx, kernel in enumerate(kernel_names):
                    if kernel == 'Baseline':
                        if bw_idx == 0:
                            baseline_value = current_data[bw_idx, kernel_idx]
                            row_data = [bandwidth, kernel]
                            for ss_idx in range(3):
                                for q_idx in range(5):
                                    row_data.append(baseline_value[ss_idx, q_idx])
                            table_data.append(row_data)
                        continue
                    
                    row_data = [bandwidth, kernel]
                    for ss_idx in range(3):
                        for q_idx in range(5):
                            row_data.append(current_data[bw_idx, kernel_idx, ss_idx, q_idx])
                    table_data.append(row_data)
            
            arrays = [
                [''] * 2 + [f'Sample Size {sample_size_names[i]}'] * 5 for i in range(3) for _ in range(5)
            ][:15]

            columns = ['Bandwidth', 'Kernel']
            for ss in sample_size_names:
                for q in quantile_names:
                    columns.append(f'SS{ss}_Q{q}')
            
            df = pd.DataFrame(table_data, columns=columns)
            all_tables.append((f"{scenario_names[scenario_idx]} - {shape_names[model_shape_idx]}", df))
    

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        title, df = all_tables[0]
        f.write(f"{title}\n")
        df.to_csv(f, index=False)

        for title, df in all_tables[1:]:
            f.write('\n\n')
            f.write(f"{title}\n")
            df.to_csv(f, index=False, header=False)
        
        f.write('\n')

# plot validation loss curves
def valloss_plot(data):

    # data shape: (scenario, model shape, kernel, sample size, quantile, 100)
    # (3, 2, 4, 3, 5, 100)

    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10

    for sceidx, scenario in enumerate(scenario_names):
        for shapeidx, shape in enumerate(shape_names):
            fig, axes = plt.subplots(5, 3, figsize=(18, 20))
            fig.suptitle(f'Validation Loss - {scenario}, {shape}', 
                        fontsize=16, fontweight='bold')
            
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            
            for qidx, q in enumerate(quantile_names):
                for nidx, n in enumerate(sample_size_names):
                    ax = axes[qidx, nidx]
                    for kidx, k in enumerate(kernel_names):
                        loss_data = data[sceidx, shapeidx, kidx, nidx, qidx, :]
                        ax.plot(loss_data, label=k, linewidth=1.5)

                    ax.set_title(f'Sample Size: {n}, Quantile: {q}')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Validation Loss')
                    ax.legend(loc='upper right', fontsize=8)

            plt.savefig(f'plots/scenario{sceidx+1}_model{shapeidx+1}_validation_loss.png', dpi=300, bbox_inches='tight')
            #plt.show()


# plot running time bar with 95% confidence interval
def time_plot(data):

    # data shape: (trial, scenario, model shape, kernel, sample size, quantile)
    # (100, 3, 2, 4, 3, 5)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    m = len(scenario_names)
    for qidx,q in enumerate(quantile_names):
        fig, axes = plt.subplots(2, m, figsize=(12, 5))
        fig.suptitle(f'Training Time - Quantile {q}', 
                    fontsize=10, y=0.98)

        n_groups = len(sample_size_names)
        n_kernels = len(kernel_names)
        group_width = 0.8
        bar_width = group_width / n_kernels

        for shapeidx, shape in enumerate(shape_names):
            for sceidx, scenario in enumerate(scenario_names):
                if m == 1:
                    ax = axes[shapeidx]
                else:
                    ax = axes[shapeidx, sceidx]
                
                X = data[:,sceidx,shapeidx,:,:,qidx]
                means = np.mean(X, axis=0)
                stds = np.std(X, axis=0)
                n = X.shape[0]
                confidence = 0.95
                t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
                conf_intervals = t_value * stds / np.sqrt(n)
                
                positions = []
                for k in range(n_groups):
                    group_positions = []
                    for l in range(n_kernels):
                        pos = k + (l - (n_kernels-1)/2) * bar_width
                        group_positions.append(pos)
                    positions.append(group_positions)
                
                bars = []
                for l in range(n_kernels):
                    bar_means = [means[l, k] for k in range(n_groups)]
                    bar_errors = [conf_intervals[l, k] for k in range(n_groups)]
                    bar_positions = [positions[k][l] for k in range(n_groups)]
                    
                    bar = ax.bar(bar_positions, bar_means, bar_width, 
                                label=kernel_names[l], color=colors[l], alpha=0.7,
                                yerr=bar_errors,
                                capsize=2.5, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
                    bars.append(bar)

                ax.set_title(f'{scenario} - {shape}', fontsize=9)
                if shapeidx == 1: 
                    ax.set_xlabel('Sample Sizes', fontsize=10)
                if sceidx == 0:
                    ax.set_ylabel('Average Running Time (seconds)', fontsize=9)
 
                ax.set_xticks(range(n_groups))
                ax.set_xticklabels([f'{N}' for N in sample_size_names])
                
                #min_value = np.min(means - conf_intervals) 
                max_value = np.max(means + conf_intervals)
                ax.set_ylim(0, max_value * 1.05)
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                if sceidx == 0 and shapeidx == 0:
                    ax.legend(fontsize=8,loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plt.savefig(f'plots/time_tau={q}.png', dpi=300, bbox_inches='tight')
        #plt.show()

# Verify Theorem 3.1 & 3.2 by plotting log(MSE) curves with respect to log(n)
def plot_loss(file_name):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from sklearn.linear_model import LinearRegression

    # .npz file with key "mse_results"
    # mse_data.shape = (50, 1, 1, 2, 5, 5)
    # (trial, scenario, bandwidth, kernel, sample size, quantile)

    dat = np.load(file_name)
    mse_data = dat['mse_results'].mean(axis=0)
    methods = ['baseline', 'conquer-gauss']
    sample_sizes = [1000, 3000, 5000, 7000, 10000]
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    mse_data = mse_data.squeeze()  # (2, 5, 5)

    # Subplots for different quantile levels
    # Two curves in each subplot (baseline & conquer)
    # y: log(MSE), x: log(sample size)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = ['blue', 'red']
    markers = ['o', 's']
    line_styles = ['-', '--']

    for i, quantile in enumerate(quantiles):
        ax = axes[i]
        for method_idx, method in enumerate(methods):
            mse_values = mse_data[method_idx, :, i]
            ax.plot(np.log(sample_sizes), np.log(mse_values), 
                    color=colors[method_idx], marker=markers[method_idx], 
                    linewidth=2, markersize=6, label=method)
            
            # Fit y by x, linear model
            X = np.array(np.log(sample_sizes)).reshape(-1, 1)
            y = np.log(mse_values)
            
            reg = LinearRegression()
            reg.fit(X, y)
            
            # Plot the fitted line
            X_fit = np.linspace(min(np.log(sample_sizes)), max(np.log(sample_sizes)), 100).reshape(-1, 1)
            y_fit = reg.predict(X_fit)
            ax.plot(X_fit, y_fit, 
                    color=colors[method_idx], linestyle='--', 
                    linewidth=1.5, alpha=0.7,
                    label=f'{method} fit (slope: {reg.coef_[0]:.3f})')
        
        ax.set_title(f'Quantile: {quantile}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Log Sample Size', fontsize=10)
        ax.set_ylabel('Log MSE', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        ax.set_xticks(np.log(sample_sizes))
        ax.set_xticklabels([f'{size:,}' for size in np.log(sample_sizes)], rotation=45)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Hide the unused subplot
    for i in range(len(quantiles), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle('MSE Comparison: Baseline vs Conquer-Gauss Across Quantiles', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('plots/MSE-samplesize-S2-A-reg.png', dpi=300, bbox_inches='tight')
    #plt.show()


def main():

    # data shape (trial,scenario,model shape,bandwidth,kernel,samplesize,quantile)

    s1_m1_stop = np.load("data/Scenario1_(5, 70)_trial50_manualgrad_stop.npz",allow_pickle=True)
    s1_m2_stop = np.load("data/Scenario1_(10, 50)_trial50_manualgrad_stop.npz",allow_pickle=True)
    s2_m1_stop = np.load("data/Scenario2_(5, 70)_trial50_manualgrad_stop.npz",allow_pickle=True)
    s2_m2_stop = np.load("data/Scenario2_(10, 50)_trial50_manualgrad_stop.npz",allow_pickle=True)
    s3_m1_stop = np.load("data/Scenario3_(5, 70)_trial50_manualgrad_stop.npz",allow_pickle=True)
    s3_m2_stop = np.load("data/Scenario3_(10, 50)_trial50_manualgrad_stop.npz",allow_pickle=True)

    time = np.zeros((50,3,2,5,4,3,5))
    time[:,0,0,:,:,:,:] = s1_m1_stop['time_results'][:,0,:,:,:,:]
    time[:,0,1,:,:,:,:] = s1_m2_stop['time_results'][:,0,:,:,:,:]
    time[:,1,0,:,:,:,:] = s2_m1_stop['time_results'][:,0,:,:,:,:]
    time[:,1,1,:,:,:,:] = s2_m2_stop['time_results'][:,0,:,:,:,:]
    time[:,2,0,:,:,:,:] = s3_m1_stop['time_results'][:,0,:,:,:,:]
    time[:,2,1,:,:,:,:] = s3_m2_stop['time_results'][:,0,:,:,:,:]

    mse = np.zeros((50,3,2,5,4,3,5))
    mse[:,0,0,:,:,:,:] = s1_m1_stop['mse_results'][:,0,:,:,:,:]
    mse[:,0,1,:,:,:,:] = s1_m2_stop['mse_results'][:,0,:,:,:,:]
    mse[:,1,0,:,:,:,:] = s2_m1_stop['mse_results'][:,0,:,:,:,:]
    mse[:,1,1,:,:,:,:] = s2_m2_stop['mse_results'][:,0,:,:,:,:]
    mse[:,2,0,:,:,:,:] = s3_m1_stop['mse_results'][:,0,:,:,:,:]
    mse[:,2,1,:,:,:,:] = s3_m2_stop['mse_results'][:,0,:,:,:,:]


    s1_m1_nostop = np.load("data/Scenario1_(5, 70)_trial50_manualgrad_nostop.npz",allow_pickle=True)
    s1_m2_nostop = np.load("data/Scenario1_(10, 50)_trial50_manualgrad_nostop.npz",allow_pickle=True)
    s2_m1_nostop = np.load("data/Scenario2_(5, 70)_trial50_manualgrad_nostop.npz",allow_pickle=True)
    s2_m2_nostop = np.load("data/Scenario2_(10, 50)_trial50_manualgrad_nostop.npz",allow_pickle=True)
    s3_m1_nostop = np.load("data/Scenario3_(5, 70)_trial50_manualgrad_nostop.npz",allow_pickle=True)
    s3_m2_nostop = np.load("data/Scenario3_(10, 50)_trial50_manualgrad_nostop.npz",allow_pickle=True)

    # plot valloss for trial 1
    valloss = np.zeros((3,2,5,4,3,5,100))    
    valloss[0,0,:,:,:,:,:] = s1_m1_nostop['val_losses'][0,0,:,:,:,:,:]
    valloss[0,1,:,:,:,:,:] = s1_m2_nostop['val_losses'][0,0,:,:,:,:,:]
    valloss[1,0,:,:,:,:,:] = s2_m1_nostop['val_losses'][0,0,:,:,:,:,:]
    valloss[1,1,:,:,:,:,:] = s2_m2_nostop['val_losses'][0,0,:,:,:,:,:]
    valloss[2,0,:,:,:,:,:] = s3_m1_nostop['val_losses'][0,0,:,:,:,:,:]
    valloss[2,1,:,:,:,:,:] = s3_m2_nostop['val_losses'][0,0,:,:,:,:,:]

    mse_nostop = np.zeros((50,3,2,5,4,3,5))
    mse_nostop[:,0,0,:,:,:,:] = s1_m1_nostop['mse_results'][:,0,:,:,:,:]
    mse_nostop[:,0,1,:,:,:,:] = s1_m2_nostop['mse_results'][:,0,:,:,:,:]
    mse_nostop[:,1,0,:,:,:,:] = s2_m1_nostop['mse_results'][:,0,:,:,:,:]
    mse_nostop[:,1,1,:,:,:,:] = s2_m2_nostop['mse_results'][:,0,:,:,:,:]
    mse_nostop[:,2,0,:,:,:,:] = s3_m1_nostop['mse_results'][:,0,:,:,:,:]
    mse_nostop[:,2,1,:,:,:,:] = s3_m2_nostop['mse_results'][:,0,:,:,:,:]

    #table_full(mse,file_path='data/mse_results.csv')
    #table_full(mse_nostop,file_path='data/mse_results_nostop.csv')
    #valloss_plot(filter_h(valloss))
    time_plot(filter_h(time))



if __name__ == "__main__":
    main()