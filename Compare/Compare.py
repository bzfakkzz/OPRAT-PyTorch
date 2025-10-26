import torch
import torch.nn.functional as F
# import mindspore.nn as nn_ms
# import mindspore.ops as ops
import mindspore as ms
import numpy as np
import os
import csv
import hashlib
import copy
import contextlib
import triton
# from collections import Counter


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

def InferAndCompareSingleModel(model, test_data, attack_data, device, path, epoch, attack, numpy_to_path=None, model_path=None):
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

    # 找出预测不一致和一致的样本（攻击成功的样本）
    diff_indices = []
    same_indices = []
    for i in range(len(attack_data)):
        if original_pred_labels[i] != attack_pred_labels[i]:
            diff_indices.append(i)
        else: same_indices.append(i)
    
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
                log_adversarial_sample(epoch, attack, i, sample_path, seed_path, attack_pred_labels[i], original_pred_labels[i],0,model_path)

        for i, sample_tensor in enumerate(attack_data_tensor[same_indices]):
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

            original_idx = same_indices[i]
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
                log_adversarial_sample(epoch, attack, i, sample_path, seed_path, attack_pred_labels[i], original_pred_labels[i],1,model_path)

    return len(diff_indices), diff_indices, len(same_indices), same_indices, new_numpy_to_path

def log_adversarial_sample(round_num, attack, sample_index, sample_path, seed_path, label1, label2,op, model_path):
    if op==0:sss='Different'
    else:sss='Same'

    log_file = os.path.join(f"{model_path}", f"{sss}/first_attack/adversarial_log.csv")
    header = ["Round", "Attack", "Sample_Index", "Sample_Path", "Seed_Path", "Seed_Label", "Sample_Label"]

    # 如果文件不存在，创建并写入表头
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # 追加记录
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([round_num, attack, sample_index, sample_path, seed_path,label1, label2])

@contextlib.contextmanager
def set_inductor_env_vars(str):
    """临时设置环境变量来影响编译行为"""
    # 保存原始环境变量
    original_env = os.environ.copy()
    
    # 设置环境变量（这些是PyTorch内部使用的）
    for i in range(len(str)):
        if i==0:
            if str[i]=='1':os.environ['INDUCTOR_FALLBACK_RANDOM'] = '1'
            else:os.environ['INDUCTOR_FALLBACK_RANDOM'] = '0'
        elif i==1:
            if str[i]=='1':os.environ['INDUCTOR_EPILOGUE_FUSION'] = '1'
            else:os.environ['INDUCTOR_EPILOGUE_FUSION'] = '0'
        elif i==2:
            if str[i]=='1':os.environ['INDUCTOR_SHAPE_PADDING'] = '1'
            else:os.environ['INDUCTOR_SHAPE_PADDING'] = '0'
        else:
            if str[i]=='1':os.environ['INDUCTOR_DYNAMIC'] = '1'
            else:os.environ['INDUCTOR_DYNAMIC'] = '0'
    
    os.environ['CC'] = 'gcc'
    os.environ['CXX'] = 'g++'

    try:
        yield
    finally:
        # 恢复原始环境变量
        os.environ.clear()
        os.environ.update(original_env)

def InferAndCompareSingleModel1(model, test_data_paths, attack_data_paths, test_data_labels, attack_data_labels,
                                type1, type2, device1, device2, f1, f2, dir_path, op):
    model.eval()
    
    type_map = {
        'fp64': torch.float64,
        'fp32': torch.float32,
        'fp16': torch.float16
    }
    
    dtype1 = type_map[type1]
    dtype2 = type_map[type2]
    
    # 创建两个模型副本
    model1 = copy.deepcopy(model)
    model2 = copy.deepcopy(model)
    
    # 转换模型精度
    model1 = model1.to(dtype1)
    model2 = model2.to(dtype2)
    
    # 应用编译优化
    with set_inductor_env_vars(f1):
        model1 = torch.compile(model1)
    
    with set_inductor_env_vars(f2):
        model2 = torch.compile(model2)
    
    # 移动到指定设备
    model1 = model1.to(device1)
    model2 = model2.to(device2)

    # 初始化结果列表
    same_indices = []
    label_change=0

    # 保存结果
    dev1_str = str(device1).replace(':', '').replace('/', '')
    dev2_str = str(device2).replace(':', '').replace('/', '')
    f_path = os.path.join(dir_path, f'details/re_attack_details_({type1}_{type2})_({dev1_str}_{dev2_str})_({f1}_{f2}).csv')
    ff_path=os.path.join(dir_path,f'label_change/label_change_({type1}_{type2})_({dev1_str}_{dev2_str})_({f1}_{f2}).csv')

    with open(f_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Indices','Seed_path','Sample_path','(Seed_label, Sample_label)','(Cur_Seed_label, Cur_Sample_label)'])

        with open(ff_path,'w',newline='',encoding='utf-8')as ff:
            wri=csv.writer(ff)
            wri.writerow(['Indices','Seed_path', 'Sample_path','(Seed_label, Sample_label)','(Cur_Seed_label, Cur_Sample_label)'])

            for i in range(len(test_data_paths)):
                test_data=np.load(test_data_paths[i])
                attack_data=np.load(attack_data_paths[i])
                
                # 转换数据精度并移动到设备
                test_tensor = torch.from_numpy(test_data).to(dtype1).to(device1)
                attack_tensor = torch.from_numpy(attack_data).to(dtype2).to(device2)
            
                # 添加批次维度，当维度是3时要补为4维
                if test_tensor.ndim == 3:
                    test_tensor = test_tensor.unsqueeze(0)
                if attack_tensor.ndim == 3:
                    attack_tensor = attack_tensor.unsqueeze(0)
                
                # 进行推理
                with torch.no_grad():
                    original_predictions = model1(test_tensor)
                    original_pred_labels = torch.argmax(original_predictions, 1)
                
                with torch.no_grad():
                    attack_predictions = model2(attack_tensor)
                    attack_pred_labels = torch.argmax(attack_predictions, 1)

                # 比较预测结果
                # op==0，统计不一致变为一致
                # op==1，统计一致变为不一致
                if op==0:
                    if(original_pred_labels.size(0)>1):
                        flag=True
                        for j in range(original_pred_labels.size(0)):
                            if original_pred_labels[j].item() != attack_pred_labels[0].item():
                                flag=False
                                break
                        if flag:
                            same_indices.append(i)
                            writer.writerow([i, test_data_paths[i], attack_data_paths[i],
                                             f'({test_data_labels[i]},{attack_data_labels[i]})',
                                             f'({original_pred_labels[0].item()}, {attack_pred_labels[0].item()}'])
                        else:
                            if(attack_pred_labels[0].item()!=attack_data_labels[i]):
                                wri.writerow([i,test_data_paths[i], attack_data_paths[i],
                                              f'({test_data_labels[i]},{attack_data_labels[i]})',
                                              f'({original_pred_labels[0].item()}, {attack_pred_labels[0].item()})'])
                                label_change+=1
                    else:
                        if original_pred_labels[0].item() == attack_pred_labels[0].item():
                            same_indices.append(i)
                            writer.writerow([i, test_data_paths[i], attack_data_paths[i],
                                             f'({test_data_labels[i]},{attack_data_labels[i]})',
                                             f'({original_pred_labels[0].item()}, {attack_pred_labels[0].item()}'])
                        else:
                            if(attack_pred_labels[0].item()!=attack_data_labels[i]):
                                wri.writerow([i,test_data_paths[i], attack_data_paths[i],
                                              f'({test_data_labels[i]},{attack_data_labels[i]})',
                                              f'({original_pred_labels[0].item()}, {attack_pred_labels[0].item()})'])
                                label_change+=1
                else:
                    if(original_pred_labels.size(0)>1):
                        flag=True
                        for j in range(original_pred_labels.size(0)):
                            if original_pred_labels[j].item() == attack_pred_labels[0].item():
                                flag=False
                                break
                        if flag:
                            same_indices.append(i)
                            writer.writerow([i, test_data_paths[i], attack_data_paths[i], 
                                             f'({test_data_labels[i]},{attack_data_labels[i]})',
                                             f'({original_pred_labels[0].item()}, {attack_pred_labels[0].item()}'])
                        else:
                            if(attack_pred_labels[0].item()!=attack_data_labels[i]):
                                wri.writerow([i,test_data_paths[i], attack_data_paths[i],
                                              f'({test_data_labels[i]},{attack_data_labels[i]})',
                                              f'({original_pred_labels[0].item()}, {attack_pred_labels[0].item()})'])
                                label_change+=1
                    else:
                        if original_pred_labels[0].item() != attack_pred_labels[0].item():
                            same_indices.append(i)
                            writer.writerow([i, test_data_paths[i], attack_data_paths[i],
                                             f'({test_data_labels[i]},{attack_data_labels[i]})',
                                             f'({original_pred_labels[0].item()}, {attack_pred_labels[0].item()}'])
                        else:
                            if(attack_pred_labels[0].item()!=attack_data_labels[i]):
                                wri.writerow([i,test_data_paths[i], attack_data_paths[i],
                                              f'({test_data_labels[i]},{attack_data_labels[i]})',
                                              f'({original_pred_labels[0].item()}, {attack_pred_labels[0].item()})'])
                                label_change+=1
                    
                # 打印进度
                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{len(test_data_paths)} samples")
    
    total_samples = len(test_data_paths)
    same_predictions = len(same_indices)
    
    print(f"Completed processing {total_samples} samples")
    print(f"Change predictions: {same_predictions} ({same_predictions/total_samples*100:.2f}%)")
    
    return total_samples, same_predictions, label_change