import torch
import torch.nn.functional as F
import mindspore.nn as nn_ms
import mindspore.ops as ops
import mindspore as ms
import numpy as np
import os
import csv


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


def InferAndCompareSingleModel(model, test_data, attack_data, device, path, epoch, attack):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    # 对原始测试数据进行推理
    with torch.no_grad():
        if isinstance(test_data, np.ndarray):
            test_data_tensor = torch.from_numpy(test_data).float().to(device)
        else:
            test_data_tensor = test_data.to(device)

        original_predictions = model(test_data_tensor)
        _, original_pred_labels = torch.max(original_predictions.data, 1)

    # 对对抗样本进行推理
    with torch.no_grad():
        if isinstance(attack_data, np.ndarray):
            attack_data_tensor = torch.from_numpy(attack_data).float().to(device)
        else:
            attack_data_tensor = attack_data.to(device)

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
        for i, sample_tensor in enumerate(attack_data_tensor):
            # 使用CPU保存以减少GPU内存占用
            sample_np = sample_tensor.cpu().numpy()

            # 命名格式: 轮次_攻击方式_样本索引.npy
            sample_path = os.path.join(
                path,
                f"round_{epoch}_attack_{attack}_sample_{i}.npy"
            )
            np.save(sample_path, sample_np)

            # 记录日志
            log_adversarial_sample(epoch, attack, i, sample_path)

    return len(diff_indices), diff_indices, original_pred_labels, attack_pred_labels


def log_adversarial_sample(round_num, attack, sample_index, sample_path):
    # 从样本路径提取模型目录
    model_dir = os.path.dirname(sample_path)
    log_file = os.path.join(model_dir, "adversarial_log.csv")
    
    header = ["Round", "Attack", "Sample_Index", "Sample_Path"]
    
    # 如果文件不存在，创建并写入表头
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    # 追加记录
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([round_num, attack, sample_index, sample_path])