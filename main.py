import random
import shutil
import time
import hashlib
import pickle
import glob

import troubleshooter as ts
from typing import List, Dict, Set, Tuple, Any
import numpy as np
import torch
import mindspore as ms
from mindspore import context

from Compare.Compare import InferAndCompare, InferAndCompareSingleModel
from Compare.Count import CountInSeq
from models import convert_weights
from attacks import generate_adversarial_samples, create_pytorch_classifier
from data.getdata import generate_random_data
import torch
import argparse

import os
import csv
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from config import model_map, NUM_CLASSES, group_0, group_1, group_2

# 辅助函数：生成numpy数组的哈希值
def numpy_to_hash(arr):
    """将numpy数组转换为哈希字符串"""
    return hashlib.md5(arr.tobytes()).hexdigest()

# 辅助函数：从已有攻击数据文件夹加载所有攻击样本
def load_existing_attack_samples(model_path):
    """从已有攻击数据文件夹加载所有攻击样本"""
    attack_samples = []
    sample_paths = []
    
    # 搜索所有攻击数据文件
    attack_pattern = os.path.join(model_path, "adversarial_samples", "round_*_attack_*.npy")
    attack_files = glob.glob(attack_pattern)
    
    for file_path in attack_files:
        try:
            sample = np.load(file_path)
            attack_samples.append(sample)
            sample_paths.append(file_path)
        except Exception as e:
            print(f"加载攻击样本文件失败 {file_path}: {e}")
    
    return attack_samples, sample_paths

def load_initial_seed_data(model_path):
    """加载初始种子数据"""
    initial_seed_path = os.path.join(model_path, "adversarial_samples", "seed_data", "initial_seed_data.npy")
    if os.path.exists(initial_seed_path):
        return np.load(initial_seed_path), initial_seed_path
    return None, None

parser = argparse.ArgumentParser(description='运行模型组')
parser.add_argument('--group', type=int, required=True, choices=[0, 1, 2], help='要运行的组号 (0, 1, 或 2)')
parser.add_argument('--gpu', type=int, required=True, help='指定使用的GPU卡号')
args = parser.parse_args()

if args.group == 0:
    models_to_run = group_0
elif args.group == 1:
    models_to_run = group_1
else:
    models_to_run = group_2

#context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
device = torch.device(f'cuda:{args.gpu}')
print(f"使用GPU设备: cuda:{args.gpu}")

#device = torch.device('cuda')
batch_size = 30

for Torch_model, input_shape in models_to_run:

    print("-----------------------------------------------------------------------------\n")
    print(f"{input_shape}模型运行中....")
    print("-----------------------------------------------------------------------------\n")

    model_path = f'PyTorch/{Torch_model}'
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(f'{model_path}/adversarial_samples', exist_ok=True)
    os.makedirs(f'{model_path}/adversarial_samples/seed_data', exist_ok=True)

    # 准备
    attack_techniques = ['FGM', 'PGD', 'CW', 'DeepFool', 'Universal']  # 'HopSkipJump'] #HopSkipJump时间太长了（需要1.5-3小时）
    
    # 加载已有的攻击数据和初始种子
    existing_attack_samples, existing_sample_paths = load_existing_attack_samples(model_path)
    initial_seed_data, initial_seed_path = load_initial_seed_data(model_path)
    
    # 如果没有初始种子数据，生成新的
    if initial_seed_data is None:
        test_data = generate_random_data(model_map[input_shape], batch_size)
        initial_seed_path = os.path.join(model_path, "adversarial_samples", "seed_data", "initial_seed_data.npy")
        np.save(initial_seed_path, test_data)
        initial_seed_data = test_data
    else:
        test_data = initial_seed_data
    
    torch_model = model_map[Torch_model](num_classes=NUM_CLASSES)
    torch_model.to(device)
    model_save_path = os.path.join(model_path, f"{Torch_model}.pth")
    torch.save(torch_model.state_dict(), model_save_path)
    execution_rounds = 100
    Robustness = []  # 鲁棒性bug列表

    # 初始化统计字典
    T = {a: 0 for a in attack_techniques}  # 不一致计数
    H = {a: 0 for a in attack_techniques}  # 历史不一致计数
    S = []  # 攻击序列
    D_diff = []  # 差异数据集
    
    # 构建种子数据池（包含初始种子和所有已有攻击数据）
    all_seed_data = [initial_seed_data]  # 保存初始种子
    all_seed_paths = [initial_seed_path]  # 所有种子数据池

    if existing_attack_samples:
        all_seed_data.extend(existing_attack_samples)
        all_seed_paths.extend(existing_sample_paths)
    
    # 将列表转换为numpy数组以便后续处理
    if len(all_seed_data) > 1:
        all_seed_data = [arr if arr.ndim == 4 else np.expand_dims(arr, axis=0) for arr in all_seed_data]
        all_seed_data = np.concatenate(all_seed_data)
    else:
        all_seed_data = all_seed_data[0]
    
    # 创建路径映射字典 - 修复索引错误
    numpy_to_path = {}
    for i, sample in enumerate(all_seed_data):
        sample_hash = numpy_to_hash(sample)
        
        # 对于初始种子数据，所有样本都映射到同一个路径
        if i < len(initial_seed_data):
            numpy_to_path[sample_hash] = initial_seed_path
        else:
            # 对于攻击数据，计算正确的索引
            attack_index = i - len(initial_seed_data)
            if attack_index < len(existing_sample_paths):
                numpy_to_path[sample_hash] = existing_sample_paths[attack_index]
            else:
                # 如果索引超出范围，使用最后一个路径
                numpy_to_path[sample_hash] = existing_sample_paths[-1] if existing_sample_paths else initial_seed_path

    print("进行初始化...")
    for attack in attack_techniques:
        print(f"现在攻击的是{attack}....")

        start = time.perf_counter()
        classifier = create_pytorch_classifier(torch_model, model_map[input_shape], NUM_CLASSES)
        attack_data = generate_adversarial_samples(attack, classifier, test_data)
        end = time.perf_counter()
        print(f"运行时间: {end - start:.6f} 秒")

        cnt, diff_indices, original_pred_labels, attack_pred_labels, new_numpy_to_path = InferAndCompareSingleModel(
            torch_model, test_data, attack_data, device, f"{model_path}/adversarial_samples", 0, attack, numpy_to_path)
        
        # 更新numpy_to_path字典
        numpy_to_path.update(new_numpy_to_path)
        
        D_new = attack_data[diff_indices]

        T[attack] = cnt
        H[attack] = 0
        
        # 将新生成的攻击数据添加到种子池中
        if len(D_new) > 0:
            # 如果种子池为空，则直接赋值；否则拼接
            if len(all_seed_data) == 0:
                all_seed_data = D_new
            else:
                all_seed_data = np.concatenate([all_seed_data, D_new])

    print("初始化结束...")

    # 初始化文件
    with open(f"{model_path}/model_robustness_stats.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Succ", "All", "Prob"])

    with open(f"{model_path}/model_robustness_details.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Index", "True Label", "Predict Label"])

    for execution_round in range(execution_rounds):
        if len(all_seed_data) <= batch_size:
            test_data = all_seed_data
            selected_indices = list(range(len(all_seed_data)))
        else:
            selected_indices = np.random.choice(len(all_seed_data), batch_size, replace=False)
            test_data = all_seed_data[selected_indices]
        
        gen = execution_round + 1
        
        print(f"这是第{gen}轮推理...")
        G = [0 for attack in attack_techniques]
        for j in range(len(attack_techniques)):
            G[j] = T[attack_techniques[j]] - H[attack_techniques[j]]
        G_max = max(G)
        C = []
        for i, g in enumerate(G):
            if g == G_max:
                C.append(i)

        F_min, c = CountInSeq(S, C, attack_techniques)
        i = random.choice(c)
        attack = attack_techniques[i]
        print(f"要进行的是{attack}攻击...")
        S.append(attack)

        classifier = create_pytorch_classifier(torch_model, model_map[input_shape], NUM_CLASSES)
        attack_data = generate_adversarial_samples(attack, classifier, test_data)

        cnt, diff_indices, original_pred_labels, attack_pred_labels, new_numpy_to_path = InferAndCompareSingleModel(
            torch_model, test_data, attack_data, device, f"{model_path}/adversarial_samples", gen, attack, numpy_to_path)
        
        # 更新numpy_to_path字典
        numpy_to_path.update(new_numpy_to_path)
        
        D_new = attack_data[diff_indices]
        cnt1 = cnt  # 成功攻击数据个数
        cnt2 = len(attack_data)  # 所有攻击数据个数

        H[attack] = T[attack]
        T[attack] = cnt
        
        # 将新生成的攻击数据添加到种子池中
        if len(D_new) > 0:
            # 如果种子池为空，则直接赋值；否则拼接
            if len(all_seed_data) == 0:
                all_seed_data = D_new
            else:
                all_seed_data = np.concatenate([all_seed_data, D_new])
        
        D_diff.extend(D_new)

        # 写入统计信息
        with open(f"{model_path}/model_robustness_stats.csv", 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([f"{gen}", cnt1, cnt2, 1.0 * cnt1 / cnt2])

        # 写入详细结果
        with open(f"{model_path}/model_robustness_details.csv", 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for i in diff_indices:
                writer.writerow([f"{gen}", i, original_pred_labels[i], attack_pred_labels[i]])

    # 保存最终的numpy到路径映射
    mapping_path = os.path.join(f"{model_path}/adversarial_samples", "numpy_to_path_mapping.pkl")
    with open(mapping_path, 'wb') as f:
        pickle.dump(numpy_to_path, f)

    print("推理轮次结束...")