import argparse
import csv
import glob
import hashlib
import logging
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 1. 设置路径
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# 2. 导入项目模块
from config import model_map, NUM_CLASSES, group_0, group_1
from data.getdata import generate_random_data
from attacks import generate_adversarial_samples, create_pytorch_classifier
from Compare.Compare import infer_and_compare_single_model, infer_and_compare_stability
from Compare.Count import CountInSeq

# --- 全局配置常量 ---
BATCH_SIZE = 30
EXECUTION_ROUNDS = 100
ATTACK_TECHNIQUES = ['FGM', 'PGD', 'CW', 'DeepFool', 'Universal']

# 配置日志
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
        
        # 路径管理 (Pathlib)
        self.base_path = Path(f'PyTorch/{self.model_name}')
        self.adv_path = self.base_path / 'adversarial_samples'
        self.seed_path = self.adv_path / 'seed_data'
        
        # 初始化
        self._init_directories()
        self.model = self._load_and_save_model()

    def _init_directories(self):
        """初始化目录结构"""
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
        """加载已有的权重，或初始化并保存新权重"""
        model = model_map[self.model_name](num_classes=NUM_CLASSES)
        weight_path = self.base_path / f"{self.model_name}.pth"
        
        # 1. 是否存在权重文件
        if weight_path.exists():
            logger.info(f"Loading existing weights from {weight_path}")
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
        else:
            logger.info(f"Initializing random weights and saving to {weight_path}")
            # 2. 如果不存在，保存当前随机初始化的权重
            torch.save(model.state_dict(), weight_path)
            
        model.to(self.device)
        return model

    @staticmethod
    def numpy_to_hash(arr: np.ndarray) -> str:
        return hashlib.md5(arr.tobytes()).hexdigest()

    def _load_data_pool(self):
        """加载数据池：初始种子 + 历史攻击样本"""
        # 1. 初始种子
        seed_file = self.seed_path / "initial_seed_data.npy"
        if seed_file.exists():
            initial_seed = np.load(seed_file)
        else:
            logger.info("生成新的随机种子数据...")
            initial_seed = generate_random_data(model_map[self.input_shape], BATCH_SIZE)
            np.save(seed_file, initial_seed)
        
        # 2. 历史样本
        attack_samples = []
        sample_paths = []
        for f in glob.glob(str(self.adv_path / "round_*_attack_*.npy")):
            try:
                attack_samples.append(np.load(f))
                sample_paths.append(f)
            except Exception as e:
                logger.error(f"Load failed {f}: {e}")

        # 3. 合并
        all_data_list = [initial_seed] + attack_samples
        processed_list = [arr if arr.ndim == 4 else np.expand_dims(arr, 0) for arr in all_data_list]
        combined_data = np.concatenate(processed_list)

        # 4. 建立映射
        path_map = {}
        # 映射种子
        for i in range(len(initial_seed)):
            h = self.numpy_to_hash(combined_data[i])
            path_map[h] = str(seed_file)
        
        # 映射样本
        curr_idx = len(initial_seed)
        for batch_idx, batch_data in enumerate(attack_samples):
            path = sample_paths[batch_idx] if batch_idx < len(sample_paths) else str(seed_file)
            for j in range(len(batch_data)):
                if curr_idx + j < len(combined_data):
                    h = self.numpy_to_hash(combined_data[curr_idx + j])
                    path_map[h] = path
            curr_idx += len(batch_data)

        return combined_data, path_map

    def _log_stats(self, category: str, round_idx: int, count: int, total: int):
        """记录 Fuzzing 统计信息"""
        csv_path = self.base_path / category / 'first_attack' / 'model_robustness_stats.csv'
        is_new = not csv_path.exists()
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["Round", "Cnt", "All", "Prob"])
            prob = count / total if total > 0 else 0
            writer.writerow([f"{round_idx}", count, total, prob])

    def run_initial_fuzzing(self):
        """阶段一：模糊测试主循环"""
        logger.info(f"=== Starting Fuzzing for {self.model_name} ===")
        
        data_pool, path_map = self._load_data_pool()
        
        # 统计追踪
        stats = {
            'T': {a: 0 for a in ATTACK_TECHNIQUES},
            'H': {a: 0 for a in ATTACK_TECHNIQUES},
            'Seq': []
        }

        # --- 初始化攻击 ---
        logger.info(">>> Phase 1: Initialization Attacks")
        for attack in ATTACK_TECHNIQUES:
            logger.info(f"Executing: {attack}")
            t0 = time.perf_counter()
            
            classifier = create_pytorch_classifier(self.model, model_map[self.input_shape], NUM_CLASSES)
            init_batch = data_pool[:BATCH_SIZE] if len(data_pool) >= BATCH_SIZE else data_pool
            
            attack_data = generate_adversarial_samples(attack, classifier, init_batch)
            logger.info(f"Time: {time.perf_counter() - t0:.4f}s")

            # 推理 & 比较
            cnt1, diff_idx, cnt2, _, new_map = infer_and_compare_single_model(
                self.model, init_batch, attack_data, self.device, 
                str(self.adv_path), 0, attack, path_map, str(self.base_path)
            )
            
            path_map.update(new_map)
            stats['T'][attack] = cnt1
            
            if len(diff_idx) > 0:
                data_pool = np.concatenate([data_pool, attack_data[diff_idx]])

        # --- Fuzzing 循环 ---
        logger.info(f">>> Phase 2: Fuzzing Loop ({EXECUTION_ROUNDS} rounds)")
        for r in range(EXECUTION_ROUNDS):
            round_num = r + 1
            logger.info(f"--- Round {round_num} ---")
            
            # 1. 采样
            if len(data_pool) <= BATCH_SIZE:
                current_batch = data_pool
            else:
                indices = np.random.choice(len(data_pool), BATCH_SIZE, replace=False)
                current_batch = data_pool[indices]

            # 2. 策略选择
            gains = [stats['T'][a] - stats['H'][a] for a in ATTACK_TECHNIQUES]
            max_gain = max(gains)
            candidates = [i for i, g in enumerate(gains) if g == max_gain]
            
            _, chosen_indices = CountInSeq(stats['Seq'], candidates, ATTACK_TECHNIQUES)
            attack_name = ATTACK_TECHNIQUES[random.choice(chosen_indices)]
            stats['Seq'].append(attack_name)
            
            logger.info(f"Strategy: {attack_name}")

            # 3. 生成
            classifier = create_pytorch_classifier(self.model, model_map[self.input_shape], NUM_CLASSES)
            attack_data = generate_adversarial_samples(attack_name, classifier, current_batch)

            # 4. 比较
            cnt1, diff_idx, cnt2, _, new_map = infer_and_compare_single_model(
                self.model, current_batch, attack_data, self.device, 
                str(self.adv_path), round_num, attack_name, path_map, str(self.base_path)
            )

            # 5. 更新
            path_map.update(new_map)
            stats['H'][attack_name] = stats['T'][attack_name]
            stats['T'][attack_name] = cnt1
            
            if len(diff_idx) > 0:
                data_pool = np.concatenate([data_pool, attack_data[diff_idx]])

            total_gen = len(attack_data)
            self._log_stats('Different', round_num, cnt1, total_gen)
            self._log_stats('Same', round_num, cnt2, total_gen)

        # 保存映射
        with open(self.adv_path / "numpy_to_path_mapping.pkl", 'wb') as f:
            pickle.dump(path_map, f)
        
        logger.info("Fuzzing Completed.")

    def run_stability_test(self, op_mode: int):
        """阶段二：稳定性/消融测试"""
        target_dir = 'Different' if op_mode == 0 else 'Same'
        counter_dir = 'Same' if op_mode == 0 else 'Different'
        
        log_file = self.base_path / target_dir / 'first_attack' / 'adversarial_log.csv'
        if not log_file.exists():
            logger.warning(f"Log not found, skipping stability test: {log_file}")
            return

        logger.info(f"=== Starting Stability Test (Target: {target_dir}) ===")
        
        # 准备目录
        re_attack_base = self.base_path / target_dir / 're_attack'
        sub_categories = ['Compile', 'Precision', 'Device']
        
        for sub in sub_categories:
            (re_attack_base / sub / 'details').mkdir(parents=True, exist_ok=True)
            (re_attack_base / sub / 'label_change').mkdir(parents=True, exist_ok=True)
            # 初始化 CSV
            self._init_stability_csv(re_attack_base / sub / 're_attack_stats.csv', counter_dir)
            self._init_label_change_csv(re_attack_base / sub / 'label_change_stats.csv')

        # 加载数据
        df = pd.read_csv(log_file)
        data_args = {
            'test_paths': df['Seed_Path'].tolist(),
            'attack_paths': df['Sample_Path'].tolist(),
            'test_labels': df['Seed_Label'].tolist(),
            'attack_labels': df['Sample_Label'].tolist()
        }

        # --- 定义测试配置 ---
        gpu_str = f'cuda:{self.gpu_id}'
        configs = []
        
        # 1. Compile 组
        for code in ['1000', '0100', '0010', '0001']:
            configs.append(('Compile', 'fp32', 'fp32', gpu_str, gpu_str, code, code))
        # 2. Device 组
        configs.append(('Device', 'fp32', 'fp32', 'cpu', 'cpu', '0000', '0000'))
        # 3. Precision 组
        configs.append(('Precision', 'fp16', 'fp16', gpu_str, gpu_str, '0000', '0000'))

        # 批量执行
        for category, pt, pa, dt, da, ct, ca in configs:
            self._run_single_stability_batch(
                category, (pt, pa, dt, da, ct, ca), 
                data_args, re_attack_base, op_mode
            )

    def _run_single_stability_batch(self, category, params, data, base_path, op_mode):
        """执行单次批次并记录"""
        pt, pa, dt, da, ct, ca = params
        
        total, target_cnt, change = infer_and_compare_stability(
            self.model,
            data['test_paths'],
            data['attack_paths'],
            data['test_labels'],
            data['attack_labels'],
            pt, pa, dt, da, ct, ca,
            str(base_path / category),
            op_mode
        )

        if total > 0:
            # 记录 re_attack_stats
            csv_path = base_path / category / 're_attack_stats.csv'
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([pt, dt, ct, total, target_cnt, target_cnt / total])

            # 记录 label_change_stats
            diff = total - target_cnt
            if diff > 0:
                csv_path = base_path / category / 'label_change_stats.csv'
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([pt, dt, ct, diff, change, change / diff])

    @staticmethod
    def _init_stability_csv(path, counter_col):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                '(Precision)', '(Device)', '(Compile)', 
                'Total_cnt', f'{counter_col}_cnt', 'Prob'
            ])

    @staticmethod
    def _init_label_change_csv(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                '(Precision)', '(Device)', '(Compile)', 
                'Still_cnt', 'label_change_cnt', 'Prob'
            ])

def main():
    # 复现性设置
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 参数解析
    parser = argparse.ArgumentParser(description='OPRAT Robustness Framework')
    parser.add_argument('--group', type=int, required=True, choices=[0, 1], help='Model Group')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--mode', type=str, choices=['fuzz', 'stability', 'all'], default='all', 
                        help='Mode: fuzz, stability, or all')
    args = parser.parse_args()

    models = group_0 if args.group == 0 else group_1
    
    logger.info(f"Task Started: GPU={args.gpu}, Mode={args.mode}")

    for model_name, input_shape in models:
        logger.info("=" * 60)
        logger.info(f"Processing Model: {model_name} ({input_shape})")
        logger.info("=" * 60)

        tester = RobustnessTester(model_name, input_shape, args.gpu)

        if args.mode in ['fuzz', 'all']:
            tester.run_initial_fuzzing()
        
        if args.mode in ['stability', 'all']:
            tester.run_stability_test(op_mode=0)
            tester.run_stability_test(op_mode=1)

if __name__ == "__main__":
    main()


