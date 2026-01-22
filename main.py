import argparse
import csv
import glob
import hashlib
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch

# 添加当前目录到环境变量，确保能引用到项目模块
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from config import model_map, NUM_CLASSES, group_0, group_1
from data.getdata import generate_random_data
from models import convert_weights
from attacks import generate_adversarial_samples, create_pytorch_classifier
from Compare.Compare import InferAndCompareSingleModel, InferAndCompareSingleModel1
from Compare.Count import CountInSeq

# --- 全局配置 ---
BATCH_SIZE = 30
EXECUTION_ROUNDS = 100
ATTACK_TECHNIQUES = ['FGM', 'PGD', 'CW', 'DeepFool', 'Universal']

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class RobustnessTester:
    def __init__(self, model_name: str, input_shape: str, gpu_id: int):
        self.model_name = model_name
        self.input_shape = input_shape
        self.device = torch.device(f'cuda:{gpu_id}')
        self.gpu_id = gpu_id
        
        # 路径配置 (使用pathlib管理)
        self.base_path = Path(f'PyTorch/{self.model_name}')
        self.adv_path = self.base_path / 'adversarial_samples'
        self.seed_path = self.adv_path / 'seed_data'
        
        # 初始化目录结构
        self._init_directories()
        
        # 加载并保存模型初始状态
        self.model = self._load_and_save_model()

    def _init_directories(self):
        """初始化所有必要的文件夹"""
        dirs = [
            self.seed_path,
            self.base_path / 'Different/first_attack',
            self.base_path / 'Different/re_attack',
            self.base_path / 'Same/first_attack',
            self.base_path / 'Same/re_attack'
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _load_and_save_model(self):
        """加载模型权重并保存一份副本"""
        model = model_map[self.model_name](num_classes=NUM_CLASSES)
        model.to(self.device)
        torch.save(model.state_dict(), self.base_path / f"{self.model_name}.pth")
        return model

    @staticmethod
    def numpy_to_hash(arr: np.ndarray) -> str:
        """生成数组的MD5哈希，用于去重和映射"""
        return hashlib.md5(arr.tobytes()).hexdigest()

    def _load_data_pool(self):
        """加载初始种子和历史攻击样本，构建数据池"""
        # 1. 加载或生成初始种子
        seed_file = self.seed_path / "initial_seed_data.npy"
        if seed_file.exists():
            initial_seed = np.load(seed_file)
        else:
            logger.info("生成初始随机种子数据...")
            initial_seed = generate_random_data(model_map[self.input_shape], BATCH_SIZE)
            np.save(seed_file, initial_seed)
        
        # 2. 加载历史攻击样本
        attack_samples = []
        sample_paths = []
        # 使用glob匹配所有round的攻击数据
        for f in glob.glob(str(self.adv_path / "round_*_attack_*.npy")):
            try:
                attack_samples.append(np.load(f))
                sample_paths.append(f)
            except Exception as e:
                logger.error(f"文件加载失败 {f}: {e}")

        # 3. 合并数据
        all_data_list = [initial_seed] + attack_samples
        # 确保维度一致 (处理可能的维度差异)
        processed_list = [arr if arr.ndim == 4 else np.expand_dims(arr, 0) for arr in all_data_list]
        combined_data = np.concatenate(processed_list)

        # 4. 重建 Hash -> Path 映射
        path_map = {}
        # 映射初始种子
        for i in range(len(initial_seed)):
            h = self.numpy_to_hash(combined_data[i])
            path_map[h] = str(seed_file)
        
        # 映射攻击样本
        curr_idx = len(initial_seed)
        for batch_idx, batch_data in enumerate(attack_samples):
            # 防止索引越界，虽然理论上不会
            path = sample_paths[batch_idx] if batch_idx < len(sample_paths) else str(seed_file)
            for j in range(len(batch_data)):
                if curr_idx + j < len(combined_data):
                    h = self.numpy_to_hash(combined_data[curr_idx + j])
                    path_map[h] = path
            curr_idx += len(batch_data)

        return combined_data, path_map

    def _log_stats(self, category: str, round_idx: int, count: int, total: int):
        """写入统计CSV"""
        csv_path = self.base_path / category / 'first_attack' / 'model_robustness_stats.csv'
        is_new = not csv_path.exists()
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["Round", "Cnt", "All", "Prob"])
            prob = count / total if total > 0 else 0
            writer.writerow([f"{round_idx}", count, total, prob])

    def run_initial_fuzzing(self):
        """第一阶段：初始模糊测试 (原代码的主循环逻辑)"""
        logger.info(f"=== 开始 {self.model_name} 的初始模糊测试 ===")
        
        data_pool, path_map = self._load_data_pool()
        
        # 状态追踪
        stats = {
            'T': {a: 0 for a in ATTACK_TECHNIQUES}, # 总次数
            'H': {a: 0 for a in ATTACK_TECHNIQUES}, # 历史次数
            'Seq': [] # 攻击序列
        }

        # --- 初始化攻击阶段 ---
        logger.info(">>> 初始化攻击阶段...")
        for attack in ATTACK_TECHNIQUES:
            logger.info(f"执行初始化攻击: {attack}")
            t0 = time.perf_counter()
            
            # 使用初始种子进行攻击
            classifier = create_pytorch_classifier(self.model, model_map[self.input_shape], NUM_CLASSES)
            # 确保只用前BATCH_SIZE个数据做初始化
            init_batch = data_pool[:BATCH_SIZE] if len(data_pool) >= BATCH_SIZE else data_pool
            
            attack_data = generate_adversarial_samples(attack, classifier, init_batch)
            logger.info(f"耗时: {time.perf_counter() - t0:.4f}s")

            # 比较推理结果
            cnt1, diff_idx, cnt2, _, new_map = InferAndCompareSingleModel(
                self.model, init_batch, attack_data, self.device, 
                str(self.adv_path), 0, attack, path_map, str(self.base_path)
            )
            
            # 更新状态
            path_map.update(new_map)
            stats['T'][attack] = cnt1
            
            # 将新生成的有效样本加入池中
            if len(diff_idx) > 0:
                data_pool = np.concatenate([data_pool, attack_data[diff_idx]])

        # --- Fuzzing 循环阶段 ---
        logger.info(f">>> 进入 Fuzzing 循环 ({EXECUTION_ROUNDS} 轮)...")
        for r in range(EXECUTION_ROUNDS):
            round_num = r + 1
            logger.info(f"--- Round {round_num} ---")
            
            # 1. 随机采样
            if len(data_pool) <= BATCH_SIZE:
                current_batch = data_pool
            else:
                indices = np.random.choice(len(data_pool), BATCH_SIZE, replace=False)
                current_batch = data_pool[indices]

            # 2. 策略选择 (基于增益 G = T - H)
            gains = [stats['T'][a] - stats['H'][a] for a in ATTACK_TECHNIQUES]
            max_gain = max(gains)
            candidates = [i for i, g in enumerate(gains) if g == max_gain]
            
            # 调用 CountInSeq 选择算法
            _, chosen_indices = CountInSeq(stats['Seq'], candidates, ATTACK_TECHNIQUES)
            attack_name = ATTACK_TECHNIQUES[random.choice(chosen_indices)]
            stats['Seq'].append(attack_name)
            
            logger.info(f"选择策略: {attack_name}")

            # 3. 生成对抗样本
            classifier = create_pytorch_classifier(self.model, model_map[self.input_shape], NUM_CLASSES)
            attack_data = generate_adversarial_samples(attack_name, classifier, current_batch)

            # 4. 推理比较
            cnt1, diff_idx, cnt2, _, new_map = InferAndCompareSingleModel(
                self.model, current_batch, attack_data, self.device, 
                str(self.adv_path), round_num, attack_name, path_map, str(self.base_path)
            )

            # 5. 更新数据与统计
            path_map.update(new_map)
            stats['H'][attack_name] = stats['T'][attack_name]
            stats['T'][attack_name] = cnt1
            
            if len(diff_idx) > 0:
                data_pool = np.concatenate([data_pool, attack_data[diff_idx]])

            total_gen = len(attack_data)
            self._log_stats('Different', round_num, cnt1, total_gen)
            self._log_stats('Same', round_num, cnt2, total_gen)

        # 保存最终映射关系
        with open(self.adv_path / "numpy_to_path_mapping.pkl", 'wb') as f:
            pickle.dump(path_map, f)
        
        logger.info("初始 Fuzzing 完成。")

    def run_stability_test(self, op_mode: int):
        """
        第二阶段：二次攻击/稳定性测试 (原 run2 函数)
        op_mode: 0 (Target=Different, Counter=Same), 1 (Target=Same, Counter=Different)
        """
        target_dir = 'Different' if op_mode == 0 else 'Same'
        counter_dir = 'Same' if op_mode == 0 else 'Different'
        
        log_file = self.base_path / target_dir / 'first_attack' / 'adversarial_log.csv'
        if not log_file.exists():
            logger.warning(f"日志文件不存在，跳过稳定性测试: {log_file}")
            return

        logger.info(f"=== 开始稳定性测试 (Target: {target_dir}) ===")
        
        # 准备输出目录
        re_attack_base = self.base_path / target_dir / 're_attack'
        sub_categories = ['Compile', 'Precision', 'Device']
        
        for sub in sub_categories:
            (re_attack_base / sub / 'details').mkdir(parents=True, exist_ok=True)
            (re_attack_base / sub / 'label_change').mkdir(parents=True, exist_ok=True)
            # 初始化CSV头
            self._init_stability_csv(re_attack_base / sub / 're_attack_stats.csv', counter_dir)
            self._init_label_change_csv(re_attack_base / sub / 'label_change_stats.csv')

        # 加载待测试数据
        df = pd.read_csv(log_file)
        # 转换为列表，方便传入 InferAndCompareSingleModel1
        data_args = {
            'seed_paths': df['Seed_Path'].tolist(),
            'sample_paths': df['Sample_Path'].tolist(),
            'seed_labels': df['Seed_Label'].tolist(),
            'sample_labels': df['Sample_Label'].tolist()
        }

        # --- 定义测试配置 (消除了原代码中的重复循环) ---
        # 格式: (类别, 精度测试, 精度攻击, 设备测试, 设备攻击, 编译测试, 编译攻击)
        gpu_str = f'cuda:{self.gpu_id}'
        configs = []
        
        # 1. Compile 组 (fp32, gpu, 不同的编译选项)
        for code in ['1000', '0100', '0010', '0001']:
            configs.append(('Compile', 'fp32', 'fp32', gpu_str, gpu_str, code, code))
            
        # 2. Device 组 (fp32, cpu, 默认编译)
        configs.append(('Device', 'fp32', 'fp32', 'cpu', 'cpu', '0000', '0000'))
        
        # 3. Precision 组 (fp16, gpu, 默认编译)
        configs.append(('Precision', 'fp16', 'fp16', gpu_str, gpu_str, '0000', '0000'))

        # 执行所有配置
        for category, pt, pa, dt, da, ct, ca in configs:
            self._run_single_stability_batch(
                category, 
                (pt, pa, dt, da, ct, ca), 
                data_args, 
                re_attack_base, 
                op_mode
            )

    def _run_single_stability_batch(self, category, params, data, base_path, op_mode):
        """执行单次稳定性对比测试并记录结果"""
        pt, pa, dt, da, ct, ca = params
        
        # 调用核心对比函数
        total, same, change = InferAndCompareSingleModel1(
            self.model,
            data['seed_paths'],
            data['sample_paths'],
            data['seed_labels'],
            data['sample_labels'],
            pt, pa, dt, da, ct, ca,
            str(base_path / category),
            op_mode
        )

        # 记录统计信息
        if total > 0:
            csv_path = base_path / category / 're_attack_stats.csv'
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([pt, dt, ct, total, same, same / total])

        # 记录标签变化
        diff = total - same
        if diff > 0:
            csv_path = base_path / category / 'label_change_stats.csv'
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([pt, dt, ct, diff, change, change / diff])

    @staticmethod
    def _init_stability_csv(path, counter_col):
        """初始化稳定性测试结果CSV头"""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                '(Precision)', '(Device)', '(Compile)', 
                'Total_cnt', f'{counter_col}_cnt', 'Prob'
            ])

    @staticmethod
    def _init_label_change_csv(path):
        """初始化标签变化统计CSV头"""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                '(Precision)', '(Device)', '(Compile)', 
                'Still_cnt', 'label_change_cnt', 'Prob'
            ])


def main():
    # --- 固定随机种子 (复现性) ---
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- 命令行参数 ---
    parser = argparse.ArgumentParser(description='模型鲁棒性测试框架')
    parser.add_argument('--group', type=int, required=True, choices=[0, 1], help='模型组 (0 或 1)')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    # 增加 mode 参数以便单独运行某个阶段
    parser.add_argument('--mode', type=str, choices=['fuzz', 'stability', 'all'], default='all', 
                        help='运行模式: fuzz (仅第一阶段), stability (仅第二阶段), all (全部)')
    args = parser.parse_args()

    # 选择模型组
    models = group_0 if args.group == 0 else group_1
    
    logger.info(f"任务启动: GPU={args.gpu}, Mode={args.mode}")

    for model_name, input_shape in models:
        logger.info("=" * 60)
        logger.info(f"处理模型: {model_name} (输入尺寸: {input_shape})")
        logger.info("=" * 60)

        tester = RobustnessTester(model_name, input_shape, args.gpu)

        # 阶段一: 初始 Fuzzing
        if args.mode in ['fuzz', 'all']:
            tester.run_initial_fuzzing()
        
        # 阶段二: 稳定性/二次攻击测试
        if args.mode in ['stability', 'all']:
            tester.run_stability_test(op_mode=0)
            tester.run_stability_test(op_mode=1)

if __name__ == "__main__":
    main()
