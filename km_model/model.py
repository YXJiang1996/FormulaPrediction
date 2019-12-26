import torch
import numpy as np
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet


# 定义神经网络模型
def model(dim_x, dim_y, dim_z):

    dim_total=max(dim_x,dim_y+dim_z)

    # 定义输入层节点
    inp = InputNode(dim_total, name='input')
