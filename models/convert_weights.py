import torch
import torch.nn as nn
import mindspore as ms
import mindspore.nn as nn_ms
import troubleshooter as ts
import os


def convert_weights(model_torch, model_ms):
    """转换并加载权重"""
    # 保存PyTorch模型权重
    os.makedirs('weights', exist_ok=True)
    torch.save(model_torch.state_dict(), 'weights/model_torch.pth')

    # 生成权重映射
    ts.migrator.get_weight_map(
        pt_net=model_torch,
        weight_map_save_path='weights/model_torch_map.json',
        print_map=False
    )

    # 转换权重
    ts.migrator.convert_weight(
        weight_map_path='weights/model_torch_map.json',
        pt_file_path='weights/model_torch.pth',
        ms_file_save_path='weights/model_ms.ckpt',
        print_conv_info=False
    )

    # 加载转换后的权重到MindSpore模型
    param_dict = ms.load_checkpoint('weights/model_ms.ckpt')
    param_not_load = ms.load_param_into_net(model_ms, param_dict)

    if param_not_load:
        print("以下参数未加载:", param_not_load)
    else:
        print("所有权重成功加载!")

    return model_torch, model_ms