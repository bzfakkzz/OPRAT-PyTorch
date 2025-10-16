import torch
import torch.nn.functional as F
import mindspore.nn as nn_ms
import mindspore.ops as ops
import mindspore as ms
import numpy as np
import os
import csv
import hashlib


# 辅助函数：生成numpy数组的哈希值
def numpy_to_hash(arr):
    """将numpy数组转换为哈希字符串"""
    return hashlib.md5(arr.tobytes()).hexdigest()


# 用device传参CPU和GPU
def InferAndCompare(torch_model, ms_model, data, device=None):
    torch_model.eval()
    if device is None:
        device = next(torch_model.parameters()).device
    torch_model.to(device)
    test_torch = torch.from_numpy(data).float().to(device)
    with torch.no_grad():
        torch_output = torch_model(test_torch)
    torch_output = torch_output.cpu().detach().numpy()

    ms_model.set_train(False)
    test_ms = ms.Tensor(data, dtype=ms.float32)
    ms_output = ms_model(test_ms).asnumpy()

    count = 0
    D_new = []

    for t, m, d in zip(torch_output, ms_output, data):
        t, m = np.asarray(t), np.asarray(m)
        chebyshev_distance = np.abs(t - m).max()
        if chebyshev_distance >= 1e-7:
            D_new.append(data)
            count = count + 1

    return count, D_new


def InferAndCompareSingleModel(model, test_data, attack_data, device, path, epoch, attack, numpy_to_path=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    
    # 统一处理输入数据：确保转换为张量并移动到正确设备
    def prepare_data(data):
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        else:
            tensor = data
        return tensor.to(device)  # 显式移动到模型所在设备
    
    # 准备测试数据和攻击数据
    test_data_tensor = prepare_data(test_data)
    attack_data_tensor = prepare_data(attack_data)
    
    # 如果没有提供numpy_to_path，创建一个空字典
    if numpy_to_path is None:
        numpy_to_path = {}
    
    # 创建新的映射字典，用于存储本轮生成的攻击数据
    new_numpy_to_path = {}

    # 对原始测试数据进行推理
    with torch.no_grad():
        original_predictions = model(test_data_tensor)
        _, original_pred_labels = torch.max(original_predictions.data, 1)

    # 对对抗样本进行推理
    with torch.no_grad():
        attack_predictions = model(attack_data_tensor)
        _, attack_pred_labels = torch.max(attack_predictions.data, 1)

    # 找出预测不一致的样本（攻击成功的样本）
    diff_indices = []
    for i in range(len(attack_data)):
        if original_pred_labels[i] != attack_pred_labels[i]:
            diff_indices.append(i)

    # 保存攻击数据张量
    if path and epoch is not None and attack:
        os.makedirs(path, exist_ok=True)
        
        for i, sample_tensor in enumerate(attack_data_tensor[diff_indices]):
            # 使用CPU保存以减少GPU内存占用
            sample_np = sample_tensor.cpu().numpy()

            # 命名格式: 轮次_攻击方式_样本索引.npy
            sample_path = os.path.join(
                path,
                f"round_{epoch}_attack_{attack}_sample_{i}.npy"
            )
            np.save(sample_path, sample_np)
            
            # 将攻击数据numpy数组映射到文件路径
            sample_hash = numpy_to_hash(sample_np)
            new_numpy_to_path[sample_hash] = sample_path

            original_idx = diff_indices[i]
            seed_np = test_data[original_idx]
            
            # 查找种子数据的路径（如果存在）
            seed_hash = numpy_to_hash(seed_np)
            seed_path = numpy_to_path.get(seed_hash, None)
            
            # 如果种子数据没有路径，使用默认路径
            if seed_path is None:
                # 使用默认的种子数据路径
                seed_path = os.path.join(path, "seed_data", "initial_seed_data.npy")
                # 如果默认路径不存在，创建一个
                if not os.path.exists(os.path.dirname(seed_path)):
                    os.makedirs(os.path.dirname(seed_path), exist_ok=True)
                # 保存种子数据
                np.save(seed_path, seed_np)
                # 将种子数据numpy数组映射到文件路径
                new_numpy_to_path[seed_hash] = seed_path
            else:
                # 如果种子数据已有路径，添加到新映射中
                new_numpy_to_path[seed_hash] = seed_path

            # 记录日志 - 只有在epoch不为0时才记录
            if epoch:
                log_adversarial_sample(epoch, attack, i, sample_path, seed_path)

    return len(diff_indices), diff_indices, original_pred_labels, attack_pred_labels, new_numpy_to_path

def log_adversarial_sample(round_num, attack, sample_index, sample_path, seed_path):
    log_file = os.path.join(os.path.dirname(sample_path), "adversarial_log.csv")
    header = ["Round", "Attack", "Sample_Index", "Sample_Path", "Seed_Path"]

    # 如果文件不存在，创建并写入表头
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # 追加记录
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([round_num, attack, sample_index, sample_path, seed_path])


def get_path_for_numpy(arr, mapping_dict):
    """根据numpy数组获取对应的文件路径"""
    arr_hash = numpy_to_hash(arr)
    return mapping_dict.get(arr_hash, None)