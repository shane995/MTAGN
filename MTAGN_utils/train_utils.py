import random
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import Dataset


def weight_inti(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def set_seed(seed=2021):
    """
    torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    torch.backends.cudnn.benchmark=True.将会让程序在开始时花费一点额外时间，
    为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    :param seed:
    :return:
    """
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.manual_seed(seed)  # 为CPU设置随机种子
    # 根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    # 但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为当前GPU设置随机种子
    # cudnn.enabled = True
    cudnn.benchmark = False  # False: 表示禁用
    cudnn.deterministic = True  # True: 每次返回的卷积算法将是确定的，即默认算法。


class MyDataset(Dataset):
    def __init__(self, x_data, y1_data, y2_data):
        self.len = x_data.shape[0]
        self.data = torch.from_numpy(x_data)
        self.labels1 = torch.from_numpy(y1_data)
        self.labels2 = torch.from_numpy(y2_data)

    def __getitem__(self, index):
        return self.data[index], self.labels1[index], self.labels2[index]

    def __len__(self):
        return self.len
