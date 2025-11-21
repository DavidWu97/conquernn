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

# 设置随机种子以确保结果可重现
np.random.seed(42)
torch.manual_seed(42)

file_path = "data/bmi"  # 替换为你的CSV文件路径
quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
kernels = ['gaussian', 'uniform', 'epanechnikov']
shape = (10,50)
bandwidths = [0.001,0.005,0.01,0.05,0.1]

def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    df = pd.read_csv(f'{file_path}/health_lifestyle_dataset.csv')
    
    # 过滤family_history为0的样本
    df = df[df['family_history'] == 0]
    
    # 选择特征和目标变量
    features = ['age', 'daily_steps', 'sleep_hours', 'water_intake_l', 'calories_consumed']
    target = 'bmi'
    
    # 按性别分组
    male_df = df[df['gender'] == 'Male']
    female_df = df[df['gender'] == 'Female']
    
    male_X = male_df[features].values
    male_y = male_df[target].values
    
    female_X = female_df[features].values
    female_y = female_df[target].values
    print(f"Sample shape for male: {male_X.shape}")
    print(f"Sample shape for female: {female_X.shape}")
    
    return male_X, male_y, female_X, female_y

def evaluate_pinball_loss_single(model, X_test, y_test, quantile):
    """评估模型在测试集上单个分位数的pinball loss"""
    predictions = model.predict(X_test)
    
    # 如果predictions是3D数组（多变量情况），调整形状
    if len(predictions.shape) == 3:
        predictions = predictions.squeeze(1)
    
    # 计算pinball loss
    loss_fn = PinballLoss(quantile=quantile)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    
    # 确保形状匹配
    if len(y_test_tensor.shape) == 1:
        y_test_tensor = y_test_tensor.unsqueeze(-1)
    
    loss = loss_fn(predictions_tensor, y_test_tensor)
    
    return loss.item()

def evaluate_pinball_loss_multi(model, X_test, y_test, quantiles):
    """评估模型在测试集上多个分位数的pinball loss"""
    predictions = model.predict(X_test)
    
    # 如果predictions是3D数组（多变量情况），调整形状
    if len(predictions.shape) == 3:
        predictions = predictions.squeeze(1)
    
    # 计算每个分位数的pinball loss
    losses = []
    for i, q in enumerate(quantiles):
        loss_fn = PinballLoss(quantile=q)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # 提取对应分位数的预测
        pred_tensor = torch.tensor(predictions[:, i], dtype=torch.float32)
        
        # 确保形状匹配
        if len(y_test_tensor.shape) == 1:
            y_test_tensor = y_test_tensor.unsqueeze(-1)
        
        loss = loss_fn(pred_tensor, y_test_tensor)
        losses.append(loss.item())
    
    return losses

def cross_validate_conquer(X_train, y_train, quantiles, kernel, k_folds=5, bandwidth_candidates=None, estimate='single'):
    """对Conquer模型进行K折交叉验证选择最优bandwidth"""
    if bandwidth_candidates is None:
        bandwidth_candidates = bandwidths
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    bandwidth_scores = {bw: [] for bw in bandwidth_candidates}
    
    print(f"开始交叉验证选择最优bandwidth (kernel={kernel}, estimate={estimate})...")
    
    for bandwidth in bandwidth_candidates:
        print(f"评估bandwidth={bandwidth}")
        
        fold_losses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"  折 {fold+1}/{k_folds}")
            
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # 训练Conquer模型
            conquer_model = ConquerNetwork(
                quantiles=quantiles,
                kernel=kernel,
                bandwidth=bandwidth,
                shape=shape,
                residual=False
            )
            
            # 训练模型
            conquer_model.fit(X_fold_train, y_fold_train)
            
            # 在验证集上评估
            if estimate == 'single':
                # 对于single模式，我们只关心第一个分位数的损失
                val_loss = evaluate_pinball_loss_single(conquer_model, X_fold_val, y_fold_val, quantiles)
            else:
                # 对于multi模式，我们计算所有分位数的平均损失
                val_losses = evaluate_pinball_loss_multi(conquer_model, X_fold_val, y_fold_val, quantiles)
                val_loss = np.mean(val_losses)
            
            fold_losses.append(val_loss)
        
        avg_loss = np.mean(fold_losses)
        bandwidth_scores[bandwidth] = avg_loss
        print(f"  bandwidth={bandwidth}, 平均Loss: {avg_loss:.4f}")
    
    # 选择最优bandwidth
    best_bandwidth = min(bandwidth_scores, key=bandwidth_scores.get)
    best_score = bandwidth_scores[best_bandwidth]
    
    print(f"最优bandwidth: {best_bandwidth}, 平均Loss: {best_score:.4f}")
    
    return best_bandwidth, bandwidth_scores

def find_matching_files(folder_path, s):
    """
    查找匹配模式 f'{s}_某个字符串_besth某个数' 的文件，并提取字符串和数字
    
    Args:
        folder_path (str): 文件夹路径
        s (str): 要匹配的字符串
    
    Returns:
        list: 包含(字符串, 数字)元组的列表
    """
    pattern = re.compile(rf"^{re.escape(s)}_(.+)_besth([\d.]+)$")
    matches = []
    
    for filename in os.listdir(folder_path):
        # 跳过子目录，只处理文件
        if os.path.isfile(os.path.join(folder_path, filename)):
            match = pattern.match(filename)
            if match:
                middle_string = match.group(1)
                number = float(match.group(2))
                matches.append((middle_string, number))
    
    return matches

# def find_matching_files(folder_path, s):
#     """
#     查找匹配 f'{s}_某个字符串_besth某个数' 模式的文件
    
#     参数:
#         folder_path: 文件夹路径
#         s: 要匹配的字符串
    
#     返回:
#         list: 包含(中间字符串, 数字)的元组列表
#     """
#     pattern = re.compile(rf'^{re.escape(s)}_(.+)_besth([-+]?(?:\d+\.?\d*|\.\d+))$')
#     matches = []
    
#     for filename in os.listdir(folder_path):
#         # 去掉文件扩展名（如果有的话）
#         name_without_ext = os.path.splitext(filename)[0]
        
#         match = pattern.match(name_without_ext)
#         if match:
#             middle_str = match.group(1)  # 提取中间字符串
#             number = float(match.group(2))  # 提取数字并转换为整数
#             matches.append((middle_str, number))
    
#     return matches

def train_and_evaluate_gender(X_train, X_test, y_train, y_test, gender, quantiles, estimate='single'):
    """为特定性别训练和评估所有模型"""
    print(f"\n处理{gender}数据...")
    
    # 存储结果
    results = defaultdict(list)
    
    if estimate == 'single':
    
        # 对于每个分位数单独训练模型
        for i, q in enumerate(quantiles):
            model_id_baseline = f"{gender}_{shape}_{estimate}q{q}_baseline"
            model_id_conquer = f"{gender}_{shape}_{estimate}q{q}_conquer"
            matches = find_matching_files(f"{file_path}/model",model_id_conquer)
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
                print(f"\n训练分位数 {q} 的模型...")
                
                # 1. Baseline模型
                print("训练Baseline模型...")
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
            
                # 2. 各种核的Conquer模型
                for kernel in kernels:
                    print(f"训练{kernel}核Conquer模型...")
                    
                    # 交叉验证选择最优bandwidth
                    best_bw, _ = cross_validate_conquer(
                        X_train, y_train, q, kernel, 
                        k_folds=5, estimate=estimate
                    )
                    
                    # 使用最优bandwidth训练最终模型
                    conquer_model = ConquerNetwork(
                        quantiles=q,
                        kernel=kernel,
                        bandwidth=best_bw,
                        shape=shape,
                        residual=False
                    )
                    conquer_model.fit(X_train, y_train)
                    
                    # 评估模型
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
            print("\n训练多分位数模型...")
            
            # 1. Baseline模型
            print("训练Baseline模型...")
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
            
            # 2. 各种核的Conquer模型
            for kernel in kernels:
                print(f"训练{kernel}核Conquer模型...")
                
                # 交叉验证选择最优bandwidth
                best_bw, _ = cross_validate_conquer(
                    X_train, y_train, quantiles, kernel, 
                    k_folds=5, estimate=estimate
                )
                
                # 使用最优bandwidth训练最终模型
                conquer_model = ConquerNetwork(
                    quantiles=quantiles,
                    kernel=kernel,
                    bandwidth=best_bw,
                    shape=shape,
                    residual=False
                )
                conquer_model.fit(X_train, y_train)
                
                # 评估模型
                conquer_losses = evaluate_pinball_loss_multi(conquer_model, X_test, y_test, quantiles)
                results[kernel] = conquer_losses
                torch.save(conquer_model,f'{file_path}/model/{model_id_conquer}_{kernel}_besth{best_bw}')
    
    return results

def create_results_table(male_results, female_results, quantiles, estimate):
    """创建结果表格"""
    methods = ['baseline', 'gaussian', 'uniform', 'epanechnikov']
    
    # 创建DataFrame
    if estimate == 'single':
        # 对于single模式，每个方法有5个值（对应5个分位数）
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
        
        # 添加性别信息
        df.insert(0, 'Gender', ['Male'] * len(methods) + ['Female'] * len(methods))
        
    else:  # estimate == 'multi'
        # 对于multi模式，每个方法有5个值（对应5个分位数）
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
        
        # 添加性别信息
        df.insert(0, 'Gender', ['Male'] * len(methods) + ['Female'] * len(methods))
    
    return df

def main():
    # 1. 加载数据
    print("加载数据...")
    male_X, male_y, female_X, female_y = load_and_preprocess_data(file_path)
    
    # 2. 设置分位数水平
    #quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    
    # 3. 划分训练集和测试集
    print("划分训练集和测试集...")
    male_X_train, male_X_test, male_y_train, male_y_test = train_test_split(
        male_X, male_y, test_size=0.2, random_state=42
    )
    
    female_X_train, female_X_test, female_y_train, female_y_test = train_test_split(
        female_X, female_y, test_size=0.2, random_state=42
    )
    
    print(f"男性样本数: 训练集 {len(male_X_train)}, 测试集 {len(male_X_test)}")
    print(f"女性样本数: 训练集 {len(female_X_train)}, 测试集 {len(female_X_test)}")
    
    # 4. 分别处理single和multi模式
    all_results = {}
    
    for estimate in ['single', 'multi']:
        print(f"\n{'='*50}")
        print(f"处理 {estimate} 模式")
        print(f"{'='*50}")
        
        # 处理男性数据
        male_results = train_and_evaluate_gender(
            male_X_train, male_X_test, male_y_train, male_y_test,
            "Male", quantiles, estimate
        )
        
        # 处理女性数据
        female_results = train_and_evaluate_gender(
            female_X_train, female_X_test, female_y_train, female_y_test,
            "Female", quantiles, estimate
        )
        
        # 创建结果表格
        results_table = create_results_table(male_results, female_results, quantiles, estimate)
        all_results[estimate] = results_table
        
        # 打印结果
        print(f"\n{estimate.upper()} 模式结果:")
        print(results_table.to_string(index=False))
        
        # 保存结果到CSV
        results_table.to_csv(f'{file_path}/results_{estimate}.csv', index=False)
    
    # 5. 可视化结果
    for estimate in ['single', 'multi']:
        df = all_results[estimate]
        
        # 重塑数据以便绘图
        plot_data = pd.melt(df, id_vars=['Gender', 'Method'], 
                           value_vars=[f'q{q}' for q in quantiles],
                           var_name='Quantile', value_name='Pinball Loss')
        
        # 提取分位数值
        plot_data['Quantile_Value'] = plot_data['Quantile'].str.replace('q', '').astype(float)
        
        plt.figure(figsize=(12, 8))
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 男性结果
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
        
        # 女性结果
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
    
    # 6. 打印最终表格
    print("\n" + "="*80)
    print("最终结果汇总")
    print("="*80)
    
    for estimate in ['single', 'multi']:
        print(f"\n{estimate.upper()} 模式:")
        print(all_results[estimate].to_string(index=False))
        print()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
    #a,b,c,d = load_and_preprocess_data('data/bmi/health_lifestyle_dataset.csv')
    #print(len(b.shape))
    