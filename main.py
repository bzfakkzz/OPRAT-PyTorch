import random
import shutil
import time

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

import os
import csv
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from config import TORCH_MODEL, MS_MODEL, model_map, NUM_CLASSES, INPUTSHAPE

# 设置随机种子确保可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
device = torch.device('cuda')
os.makedirs('adversarial_samples', exist_ok=True)
os.makedirs('adversarial_samples/seed_data', exist_ok=True)
os.makedirs('model_robustness', exist_ok=True)
os.makedirs('model_robustness/pytorch', exist_ok=True)

# 基础目录
base_ad_dir = "adversarial_samples"
base_seed_dir = "adversarial_samples/seed_data"
base_robustness_dir = "model_robustness/pytorch"

for Torch_model, Mindspore_model, input_shape in zip(TORCH_MODEL, MS_MODEL, INPUTSHAPE):
    # 为当前模型创建专属目录
    model_name = Torch_model  # 使用模型名称作为目录名
    ad_dir = os.path.join(base_ad_dir, model_name)
    seed_dir = os.path.join(base_seed_dir, model_name)
    robustness_dir = os.path.join(base_robustness_dir, model_name)
    
    # 创建模型专属目录
    os.makedirs(ad_dir, exist_ok=True)
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(robustness_dir, exist_ok=True)
    
    # 创建模型专属日志文件
    attack_info_file = os.path.join(ad_dir, "attack_generation_info.txt")
    with open(attack_info_file, 'w') as info_file:
        info_file.write("Generation\tAttack\tSeed_Data_Path\n")
    
    # 创建模型专属结果文件
    robustness_stats_file = os.path.join(robustness_dir, "robustness_stats.csv")
    robustness_details_file = os.path.join(robustness_dir, "robustness_details.csv")
    
    with open(robustness_stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Succ", "All", "Prob"])
    
    with open(robustness_details_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Index", "True Label", "Predict Label"])
    
    # 模型专属对抗样本目录
    model_ad_dir = ad_dir
    
    print("-----------------------------------------------------------------------------\n")
    print(f"{model_name}模型运行中....")
    print("-----------------------------------------------------------------------------\n")

    # 准备
    attack_techniques = ['FGM', 'PGD', 'CW', 'DeepFool', 'Universal']
    test_data = generate_random_data(model_map[input_shape], 10)
    torch_model = model_map[Torch_model](num_classes=NUM_CLASSES)
    torch_model.cuda(device)
    execution_rounds = 100
    Robustness = []

    # 保存初始种子数据
    initial_seed_path = os.path.join(seed_dir, "initial_seed_data.npy")
    np.save(initial_seed_path, test_data.numpy())

    # 记录初始种子数据信息
    with open(attack_info_file, 'a') as info_file:
        info_file.write(f"initial\t-\t{initial_seed_path}\n")

    # 初始化统计字典
    T = {a: 0 for a in attack_techniques}
    H = {a: 0 for a in attack_techniques}
    S = []
    D_diff = []

    print("进行初始化...")
    for attack in attack_techniques:
        print(f"现在攻击的是{attack}....")
        start = time.perf_counter()
        classifier = create_pytorch_classifier(torch_model, model_map[input_shape], NUM_CLASSES)
        attack_data = generate_adversarial_samples(attack, classifier, test_data)
        end = time.perf_counter()
        print(f"运行时间: {end - start:.6f} 秒")

        cnt, diff_indices, original_pred_labels, attack_pred_labels = InferAndCompareSingleModel(
            torch_model, test_data, attack_data, device, model_ad_dir, 0, attack)
        D_new = attack_data[diff_indices]

        T[attack] = cnt
        H[attack] = 0
        D_diff.extend(D_new)

        with open(attack_info_file, 'a') as info_file:
            info_file.write(f"0\t{attack}\t{initial_seed_path}\n")
    print("初始化结束...")

    for execution_round in range(execution_rounds):
        test_data = generate_random_data(model_map[input_shape], 10)
        gen = execution_round + 1
        seed_path = os.path.join(seed_dir, f"gen_{gen}_seed_data.npy")
        np.save(seed_path, test_data.numpy())

        with open(attack_info_file, 'a') as info_file:
            info_file.write(f"{gen}\t-\t{seed_path}\n")

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

        cnt, diff_indices, original_pred_labels, attack_pred_labels = InferAndCompareSingleModel(
            torch_model, test_data, attack_data, device, model_ad_dir, gen, attack)
        D_new = attack_data[diff_indices]
        cnt1 = cnt
        cnt2 = len(attack_data)

        H[attack] = T[attack]
        T[attack] = cnt
        D_diff.extend(D_new)

        with open(attack_info_file, 'a') as info_file:
            info_file.write(f"{gen}\t{attack}\t{seed_path}\n")

        with open(robustness_stats_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([f"{gen}", cnt1, cnt2, 1.0 * cnt1 / cnt2])

        with open(robustness_details_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for i in diff_indices:
                writer.writerow([f"{gen}", i, original_pred_labels[i], attack_pred_labels[i]])

    print("推理轮次结束...")
    
    # for execution_round in range(execution_rounds):
    #     print(f"这是第{execution_round+1}轮推理...")
    #     G=[0 for attack in attack_techniques]
    #     for j in range(len(attack_techniques)):
    #         G[j]=T[attack_techniques[j]]-H[attack_techniques[j]]
    #     # G_max ← max(G); C ←{j|G[j]=Gmax}
    #     G_max=max(G)
    #     C=[]
    #     for i,g in enumerate(G):
    #         if g==G_max:
    #             C.append(i)

    #     # F_min ←min{ CountInSeq(S,aj) | je C}
    #     # C'←{j C | CountInSeq(S,a¡) = F_min }
    #     F_min,c=CountInSeq(S,C,attack_techniques)
    #     #i<-randomchoice(C');ar←F_min[i]
    #     i=random.choice(c)
    #     attack=attack_techniques[i]
    #     print(f"要进行的是{attack}攻击...")
    #     #add ar to S
    #     S.append(attack)

    #     #D_attack ←- DataAttack(D_test, aj)
    #     classifier = create_pytorch_classifier(torch_model, model_map[input_shape], NUM_CLASSES)
    #     attack_data = generate_adversarial_samples(attack, classifier, test_data)

    #     #t, 𝒟_new ← InferAndCompare(M₁, M₂, 𝒟_attack)
    #     t,D_new=InferAndCompare(torch_model,ms_model,attack_data)

    #     #H[a,]← T[a,]; T[ar]← tnew
    #     H[attack]=T[attack]
    #     T[attack]=t
    #     #D_diff ← D_diff U D_new
    #     D_diff.extend(D_new)

    #     torch_diff, ms_diff, torch_true_labels, torch_pred_labels, ms_true_labels, ms_pred_labels = GetCmp(
    #         torch_model, ms_model, test_data, attack_data)
    #     with open("model_robustness/pytorch_robustness_test.csv",'w',newline='',encoding='utf-8')as f:
    #         writer=csv.writer(f)
    #         for i in torch_diff:
    #             writer.writerow([f"{execution_round+1}", torch_true_labels[i], torch_pred_labels[i]])
    #     with open("model_robustness/mindspore_robustness_test.csv",'w',newline='',encoding='utf-8')as f:
    #         writer=csv.writer(f)
    #         for i in torch_diff:
    #             writer.writerow([f"{execution_round+1}", ms_true_labels[i], ms_pred_labels[i]])
            
    # print("推理轮次结束...")
    # with open('out.txt', 'w', encoding='utf-8') as f:
    #     for item in D_diff:
    #         f.write(f"{item}\n")

    # print("层次比较")
    # os.makedirs(f"{input_shape}_output",exist_ok=True)
    # os.makedirs(f"{input_shape}_results",exist_ok=True)
    # count=0
    # for input in D_diff:
    #     print(f"第{count+1}组比较中...")
    #     if count%10!=0:
    #         count=count+1
    #         continue   #内存不够了，只能每10轮记录一次
    #     #torch层输出
    #     torch_model.eval()
    #     test_torch = torch.from_numpy(input).float()
    #     ts.migrator.api_dump_init(
    #         torch_model,
    #         output_path=f"{input_shape}_output/torch_test_dump{count}",
    #         retain_backward=False
    #     )
    #     with torch.no_grad():
    #         ts.migrator.api_dump_start()
    #         torch_output = torch_model(test_torch)
    #         ts.migrator.api_dump_stop()

    #     #mindspore层输出
    #     ms_model.set_train(False)
    #     test_ms = ms.Tensor(input, dtype=ms.float32)
    #     ts.migrator.api_dump_init(
    #         ms_model,
    #         output_path=f"{input_shape}_output/ms_test_dump{count}",
    #         retain_backward=False
    #     )
    #     ts.migrator.api_dump_start()
    #     ms_output = ms_model(test_ms)
    #     ts.migrator.api_dump_stop()

    #     # 使用 TroubleShooter 的比较功能
    #     ts.migrator.api_dump_compare(
    #         f'{input_shape}_output/ms_test_dump{count}',
    #         f'{input_shape}_output/torch_test_dump{count}',
    #         output_path=f'{input_shape}_results/comparison_results{count}'
    #     )

    #     shutil.rmtree(f'{input_shape}_output/ms_test_dump{count}')
    #     shutil.rmtree(f'{input_shape}_output/torch_test_dump{count}')
    #     count=count+1

    # shutil.rmtree(f'{input_shape}_output')