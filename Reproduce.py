import os
import argparse
import numpy as np
import torch
import csv
import sys
import importlib

# 将当前目录添加到sys.path，以便导入config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config import model_map, NUM_CLASSES
except ImportError:
    print("警告: 无法导入config.py，将使用默认配置")
    model_map = {}
    NUM_CLASSES = 1000

def parse_arguments():
    parser = argparse.ArgumentParser(description='复现鲁棒性问题')

    # 必需参数
    parser.add_argument('--adv_sample_path', type=str, required=True,
                        help='对抗样本文件路径')
    parser.add_argument('--seed_sample_path', type=str, required=True,
                        help='种子数据文件路径')
    parser.add_argument('--model_type', type=str, required=True,
                        help='模型类型（必须与生成对抗样本时使用的模型类型一致，如TorchDenseNet）')

    # 可选参数
    parser.add_argument('--model_path', type=str, default="",
                        help='模型权重文件路径（默认为模型类型对应的路径）')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--input_shape', type=str, default='',
                        help='输入数据形状，格式为"通道,高,宽"（如果未提供，则尝试从config中获取）')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                        help=f'分类类别数量（默认: {NUM_CLASSES})')

    return parser.parse_args()


def load_model(model_path, model_type, num_classes, device):
    """加载指定模型"""
    # 尝试从config.py的model_map中获取模型类
    if model_type in model_map:
        model_class = model_map[model_type]
        print(f"从config.py加载模型类: {model_type}")
    else:
        # 尝试动态导入模型
        try:
            # 模型类型格式应为"模块名.类名"或"类名"
            if '.' in model_type:
                module_name, class_name = model_type.rsplit('.', 1)
            else:
                module_name = 'models'
                class_name = model_type
                
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            print(f"动态加载模型类: {model_type}")
        except (ImportError, AttributeError):
            raise ValueError(f"无法加载模型类: {model_type}。请确保模型类型正确且在config.py中定义")

    # 创建模型实例
    model = model_class(num_classes=num_classes)
    
    # 加载模型权重
    if os.path.exists(model_path):
        # 使用map_location确保在不同设备上加载
        state_dict = torch.load(model_path, map_location=device)
        
        # 检查键是否匹配
        model_keys = set(model.state_dict().keys())
        state_keys = set(state_dict.keys())
        
        if model_keys != state_keys:
            print(f"警告: 权重键不匹配 (模型: {len(model_keys)}, 文件: {len(state_keys)})")
            print(f"模型缺少的键: {model_keys - state_keys}")
            print(f"权重文件多余的键: {state_keys - model_keys}")
            
            # 尝试加载匹配的键
            matched_state_dict = {}
            for name, param in model.state_dict().items():
                if name in state_dict:
                    matched_state_dict[name] = state_dict[name]
                else:
                    print(f"警告: 键 '{name}' 在权重文件中不存在，使用随机初始化")
                    matched_state_dict[name] = param
            
            model.load_state_dict(matched_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
        
        print(f"已加载模型权重: {model_path}")
    else:
        print(f"警告: 未找到模型权重文件 {model_path}，使用随机初始化的模型")

    model.to(device)
    model.eval()
    return model


def load_sample_data(adv_path, seed_path, input_shape):
    """加载样本数据"""
    # 解析输入形状
    try:
        channels, height, width = map(int, input_shape.split(','))
    except:
        raise ValueError("输入形状格式错误，应为'通道,高,宽'，例如'3,224,224'")

    # 加载对抗样本
    adv_data = np.load(adv_path)
    if adv_data.shape != (channels, height, width):
        print(f"警告: 对抗样本形状为{adv_data.shape}，但预期为{(channels, height, width)}")

    # 加载种子数据
    seed_data = np.load(seed_path)
    if seed_data.shape != (channels, height, width):
        print(f"警告: 种子数据形状为{seed_data.shape}，但预期为{(channels, height, width)}")

    return seed_data, adv_data


def run_inference(model, data, device):
    """运行模型推理"""
    if isinstance(data, np.ndarray):
        # 添加批次维度并转换为张量
        data_tensor = torch.from_numpy(data).float().to(device).unsqueeze(0)
    else:
        data_tensor = data.to(device).unsqueeze(0)

    with torch.no_grad():
        predictions = model(data_tensor)
        probabilities = torch.softmax(predictions, dim=1)
        _, pred_label = torch.max(predictions.data, 1)

    return pred_label.item(), probabilities.cpu().numpy()[0]


def get_input_shape_from_config(model_type):
    """尝试从config.py获取输入形状"""
    # 查找与模型类型对应的输入形状键
    for key, value in model_map.items():
        if isinstance(value, tuple) and model_type in key:
            return value
    
    # 如果找不到，尝试从模型类型推断
    if "VGG" in model_type:
        return (3, 224, 224)
    elif "AlexNet" in model_type:
        return (3, 32, 32)
    elif "DenseNet" in model_type:
        return (3, 224, 224)
    elif "Inception" in model_type:
        return (64, 224, 224)
    elif "LeNet" in model_type:
        return (1, 28, 28)
    elif "LSTM" in model_type:
        return (32, 20, 10)  # 默认LSTM输入形状
    elif "MobileNet" in model_type:
        return (3, 224, 224)
    elif "ResNet" in model_type:
        return (3, 224, 224)
    elif "Xception" in model_type:
        return (3, 299, 299)
    
    return None


def reproduce_robustness_issue():
    args = parse_arguments()
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 设置默认模型路径
    if not args.model_path:
        args.model_path = f"PyTorch/{args.model_type}/{args.model_type}.pth"
        print(f"使用默认模型路径: {args.model_path}")

    # 获取输入形状
    if args.input_shape:
        input_shape = args.input_shape
    else:
        # 尝试从config.py获取输入形状
        config_shape = get_input_shape_from_config(args.model_type)
        if config_shape:
            input_shape = f"{config_shape[0]},{config_shape[1]},{config_shape[2]}"
            print(f"从config.py获取输入形状: {input_shape}")
        else:
            raise ValueError("未提供输入形状，且无法从config.py自动获取，请使用--input_shape指定")

    # 加载模型
    print(f"加载模型: {args.model_type} (分类数: {args.num_classes})")
    model = load_model(args.model_path, args.model_type, args.num_classes, device)

    # 加载样本数据
    print(f"加载对抗样本: {args.adv_sample_path}")
    print(f"加载种子数据: {args.seed_sample_path}")
    seed_data, adv_data = load_sample_data(
        args.adv_sample_path,
        args.seed_sample_path,
        input_shape
    )

    # 运行推理
    print("\n运行推理...")
    seed_label, seed_probs = run_inference(model, seed_data, device)
    adv_label, adv_probs = run_inference(model, adv_data, device)

    # 显示结果
    print("\n=== 复现结果 ===")
    print(f"种子数据预测标签: {seed_label}")
    print(f"对抗样本预测标签: {adv_label}")

    if seed_label != adv_label:
        print("\n✅ 成功复现鲁棒性问题: 原始预测和对抗预测不一致!")
        print(f"不一致程度: {abs(seed_label - adv_label)}")
    else:
        print("\n❌ 未能复现鲁棒性问题: 原始预测和对抗预测一致")

    # 显示置信度
    print(f"\n种子数据预测置信度: {seed_probs[seed_label]:.4f}")
    print(f"对抗样本预测置信度: {adv_probs[adv_label]:.4f}")

    # 显示top-5预测结果
    print("\n种子数据Top-5预测:")
    top5_indices = np.argsort(seed_probs)[::-1][:5]
    for i, idx in enumerate(top5_indices):
        print(f"{i + 1}. 类别 {idx}: {seed_probs[idx]:.4f}")

    print("\n对抗样本Top-5预测:")
    top5_indices = np.argsort(adv_probs)[::-1][:5]
    for i, idx in enumerate(top5_indices):
        print(f"{i + 1}. 类别 {idx}: {adv_probs[idx]:.4f}")


if __name__ == '__main__':
    reproduce_robustness_issue()