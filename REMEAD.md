**主要内容在main中**

运行: python main.py

伪代码:

![img.png](img.png)


**实验结果说明：**


运行结果全部存放在PyTorch文件夹中。

1. 当前TorchLeNet5和TorchAlexNetCIFAR10跑出了部分torch.compile()实验结果
2. 二次攻击数据全部存放在re_attack文件夹中，
其中re_attack_stats.csv存放了消融实验的实验结果，
Compile文件夹存放了针对torch.compile()的具体实验情况（但是没有记录label，已经在代码中更正），Device和Precision以此类推