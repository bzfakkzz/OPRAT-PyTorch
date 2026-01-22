import csv
import contextlib
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import copy

# 日志
logger = logging.getLogger(__name__)

def numpy_to_hash(arr: np.ndarray) -> str:
    """计算 Numpy 数组的 MD5 哈希值，用于去重"""
    return hashlib.md5(arr.tobytes()).hexdigest()

@contextlib.contextmanager
def set_inductor_env_vars(config_str: str):
    """
    临时设置 PyTorch Inductor 编译选项环境变量。
    config_str: 4位字符串 (e.g., '1000') 分别对应:
      0: FALLBACK_RANDOM
      1: EPILOGUE_FUSION
      2: SHAPE_PADDING
      3: DYNAMIC
    """
    env_keys = [
        'INDUCTOR_FALLBACK_RANDOM',
        'INDUCTOR_EPILOGUE_FUSION',
        'INDUCTOR_SHAPE_PADDING',
        'INDUCTOR_DYNAMIC'
    ]
    
    # 保存原始环境变量
    original_env = os.environ.copy()
    
    # 设置新环境变量
    for idx, key in enumerate(env_keys):
        if idx < len(config_str):
            os.environ[key] = '1' if config_str[idx] == '1' else '0'
    
    # 强制使用 GCC (视服务器环境可选)
    os.environ['CC'] = 'gcc'
    os.environ['CXX'] = 'g++'

    try:
        yield
    finally:
        # 恢复原始环境变量
        os.environ.clear()
        os.environ.update(original_env)

def _save_adversarial_sample(
    data_tensor: torch.Tensor, 
    path: Path, 
    file_name: str, 
    numpy_to_path: Dict[str, str],
    seed_np: np.ndarray
) -> Tuple[str, str, str, str]:
    """辅助函数：保存单个对抗样本及其对应的种子映射"""
    path.mkdir(parents=True, exist_ok=True)
    
    # 保存攻击样本
    sample_np = data_tensor.cpu().numpy()
    sample_file_path = path / file_name
    np.save(sample_file_path, sample_np)
    
    # 计算哈希
    sample_hash = numpy_to_hash(sample_np)
    seed_hash = numpy_to_hash(seed_np)
    
    # 获取种子路径 (如果映射中没有，则指向默认种子路径)
    seed_file_path = numpy_to_path.get(seed_hash)
    
    if not seed_file_path:
        # Fallback: 如果找不到种子路径，保存到默认位置
        default_seed_path = path / "seed_data" / "initial_seed_data.npy"
        if not default_seed_path.parent.exists():
            default_seed_path.parent.mkdir(parents=True, exist_ok=True)
        if not default_seed_path.exists():
             np.save(default_seed_path, seed_np)
        seed_file_path = str(default_seed_path)
    
    return str(sample_file_path), str(seed_file_path), sample_hash, seed_hash

def _log_adversarial_event(
    base_path: str, 
    category: str, 
    round_num: int, 
    attack: str, 
    idx: int, 
    sample_path: str, 
    seed_path: str, 
    pred_label: int, 
    true_label: int
):
    """辅助函数：记录对抗样本生成日志"""
    log_file = Path(base_path) / category / "first_attack" / "adversarial_log.csv"
    file_exists = log_file.exists()
    
    # 确保目录存在
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Round", "Attack", "Sample_Index", "Sample_Path", "Seed_Path", "Seed_Label", "Sample_Label"])
        writer.writerow([round_num, attack, idx, sample_path, seed_path, true_label, pred_label])

def infer_and_compare_single_model(
    model: torch.nn.Module, 
    test_data: np.ndarray, 
    attack_data: np.ndarray, 
    device: torch.device, 
    save_dir: str, 
    epoch: int, 
    attack_name: str, 
    numpy_to_path: Dict[str, str] = None, 
    base_model_path: str = None
):
    """
    第一阶段核心：推理并筛选出攻击成功/失败的样本，分别保存。
    """
    model.eval()
    model.to(device)
    
    if numpy_to_path is None:
        numpy_to_path = {}
    
    new_path_map = {}
    save_path = Path(save_dir)

    # 数据转 Tensor
    def to_tensor(data):
        return torch.from_numpy(data).float().to(device)

    test_tensor = to_tensor(test_data)
    attack_tensor = to_tensor(attack_data)

    # 批量推理
    with torch.no_grad():
        orig_preds = model(test_tensor).argmax(dim=1)
        attack_preds = model(attack_tensor).argmax(dim=1)

    # 区分样本
    # diff: 预测改变 (攻击成功)
    # same: 预测未变 (攻击失败)
    diff_mask = (orig_preds != attack_preds)
    
    diff_indices = torch.where(diff_mask)[0].cpu().numpy()
    same_indices = torch.where(~diff_mask)[0].cpu().numpy()

    # 保存与日志记录 (仅在非测试轮次执行)
    if save_dir and epoch is not None and attack_name:
        
        # 处理 Different 组
        for idx in diff_indices:
            idx = int(idx)
            f_name = f"round_{epoch}_attack_{attack_name}_sample_{idx}.npy"
            
            s_path, seed_path, s_hash, seed_hash = _save_adversarial_sample(
                attack_tensor[idx], save_path, f_name, numpy_to_path, test_data[idx]
            )
            
            new_path_map[s_hash] = s_path
            new_path_map[seed_hash] = seed_path
            
            if epoch > 0:
                _log_adversarial_event(base_model_path, 'Different', epoch, attack_name, idx, 
                                     s_path, seed_path, attack_preds[idx].item(), orig_preds[idx].item())

        # 处理 Same 组
        for idx in same_indices:
            idx = int(idx)
            f_name = f"round_{epoch}_attack_{attack_name}_sample_{idx}.npy"
            
            s_path, seed_path, s_hash, seed_hash = _save_adversarial_sample(
                attack_tensor[idx], save_path, f_name, numpy_to_path, test_data[idx]
            )
            
            new_path_map[s_hash] = s_path
            new_path_map[seed_hash] = seed_path
            
            if epoch > 0:
                _log_adversarial_event(base_model_path, 'Same', epoch, attack_name, idx, 
                                     s_path, seed_path, attack_preds[idx].item(), orig_preds[idx].item())

    return len(diff_indices), diff_indices, len(same_indices), same_indices, new_path_map

def infer_and_compare_stability(
    model: torch.nn.Module, 
    test_paths: List[str], 
    attack_paths: List[str], 
    test_labels: List[int], 
    attack_labels: List[int],
    prec_test: str, prec_attack: str, 
    dev_test: str, dev_attack: str, 
    comp_test: str, comp_attack: str, 
    log_dir: str, 
    op_mode: int
):
    """
    第二阶段核心：稳定性/消融实验对比。
    op_mode=0: Target Different (统计不一致 -> 一致)
    op_mode=1: Target Same (统计一致 -> 一致)
    """
    dtype_map = {'fp64': torch.float64, 'fp32': torch.float32, 'fp16': torch.float16}
    dtype1 = dtype_map[prec_test]
    dtype2 = dtype_map[prec_attack]

    # 准备模型副本
    model1 = copy.deepcopy(model).to(dtype1)
    model2 = copy.deepcopy(model).to(dtype2)

    # 应用编译优化
    with set_inductor_env_vars(comp_test):
        model1 = torch.compile(model1)
    with set_inductor_env_vars(comp_attack):
        model2 = torch.compile(model2)
        
    model1.to(dev_test).eval()
    model2.to(dev_attack).eval()

    # 准备日志
    log_dir_path = Path(log_dir)
    dev_suffix = f"({dev_test}_{dev_attack})".replace(':', '').replace('/', '')
    param_suffix = f"({prec_test}_{prec_attack})_{dev_suffix}_({comp_test}_{comp_attack})"
    
    detail_csv = log_dir_path / "details" / f"re_attack_details_{param_suffix}.csv"
    change_csv = log_dir_path / "label_change" / f"label_change_{param_suffix}.csv"

    # 确保目录存在
    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    change_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_header = ['Indices', 'Seed_path', 'Sample_path', '(Seed_label, Sample_label)', '(Cur_Seed_label, Cur_Sample_label)']
    
    with open(detail_csv, 'w', newline='', encoding='utf-8') as f_det, \
         open(change_csv, 'w', newline='', encoding='utf-8') as f_chg:
        
        writer_det = csv.writer(f_det)
        writer_chg = csv.writer(f_chg)
        writer_det.writerow(csv_header)
        writer_chg.writerow(csv_header)

        target_count = 0
        label_change_count = 0
        total_samples = len(test_paths)

        for i in range(total_samples):
            try:
                # 加载并处理数据
                seed_data = np.load(test_paths[i])
                sample_data = np.load(attack_paths[i])
                
                t1 = torch.from_numpy(seed_data).to(dtype1).to(dev_test)
                t2 = torch.from_numpy(sample_data).to(dtype2).to(dev_attack)
                
                if t1.ndim == 3: t1 = t1.unsqueeze(0)
                if t2.ndim == 3: t2 = t2.unsqueeze(0)

                with torch.no_grad():
                    pred1 = model1(t1).argmax(dim=1).item()
                    pred2 = model2(t2).argmax(dim=1).item()

                row_data = [
                    i, test_paths[i], attack_paths[i],
                    f'({test_labels[i]},{attack_labels[i]})',
                    f'({pred1}, {pred2})'
                ]

                # 核心统计逻辑
                is_consistent = (pred1 == pred2)
                
                if op_mode == 0:
                    # Target: Different. 关注有多少变成了 Same (Consistent)
                    if is_consistent:
                        target_count += 1
                        writer_det.writerow(row_data)
                    # 如果依然 Different，但 Error Label 变了，记为 Label Change
                    elif pred2 != attack_labels[i]:
                        label_change_count += 1
                        writer_chg.writerow(row_data)
                        
                else:
                    # Target: Same. 关注有多少依然是 Same (Consistent)
                    if is_consistent:
                        target_count += 1
                        writer_det.writerow(row_data)
                    # 如果变成了 Different，且 Error Label 变了
                    elif pred2 != attack_labels[i]:
                        label_change_count += 1
                        writer_chg.writerow(row_data)

            except Exception as e:
                logger.error(f"Sample {i} processing error: {e}")
                continue
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{total_samples} samples...")

    return total_samples, target_count, label_change_count
