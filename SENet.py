import torch
import torch.nn as nn
from torch.nn import functional as F

from convolution import *
from KANConv import *
from KANLinear import *

batch_size = 8


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,bias=True)  # 如果设为 False，model_segnet_mtan.py 91行的 nn.init.constant_(m.bias, 0) 需要注释掉
        # self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # 深度可分离卷积
        self.fc1dws = DepthwiseSeparableConv(channels, channels // 2)
        self.fc2dws = DepthwiseSeparableConv(channels, channels // 4)
        self.fc3dws = DepthwiseSeparableConv(channels, channels // 8)
        self.fc4dws = DepthwiseSeparableConv(channels, channels // reduction)

        # 1*1 Conv代替线性层
        self.linear1 = nn.Conv2d(channels // 2, channels, kernel_size=1, bias=True)
        self.linear2 = nn.Conv2d(channels // 4, channels, kernel_size=1, bias=True)
        self.linear3 = nn.Conv2d(channels // 8, channels, kernel_size=1, bias=True)
        self.linear4 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True)
        
        # KConv
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.KanCA = KAN_Convolutional_Layer(n_convs = 1,padding = (1,1),kernel_size= (3,3),device = device)
        # self.KanCA = KAN_Convolutional_Layer(n_convs = 1,padding = (0,0),kernel_size= (1,1),device = device)
        
        # self.KanCA_1 = KAN_Convolutional_Layer(n_convs = 1,padding = (0,0),kernel_size= (1,1),device = device)
        # self.KanCA_2 = KAN_Convolutional_Layer(n_convs = 1,padding = (0,0),kernel_size= (1,1),device = device)
        # self.KanCA_3 = KAN_Convolutional_Layer(n_convs = 1,padding = (0,0),kernel_size= (1,1),device = device)
        # self.KanCA_4 = KAN_Convolutional_Layer(n_convs = 1,padding = (0,0),kernel_size= (1,1),device = device)

    def forward(self, x):
        # module_input = x
        x1 = x
        x2 = x
        x3 = x
        x4 = x

        x = self.fc1(x)
        x = self.KanCA(x)
        # x = self.relu(x)
        x = self.gelu(x)
        x = self.fc2(x)

        x1 = self.fc1dws(x1)
        x1 = self.avg_pool(x1)
        x1 = self.gelu(x1)
        # x1 = self.KanCA_1(x1)
        x1 = self.linear1(x1)

        x2 = self.fc2dws(x2)
        x2 = self.avg_pool(x2)
        x2 = self.gelu(x2)
        # x2 = self.KanCA_2(x2)
        x2 = self.linear2(x2)

        x3 = self.fc3dws(x3)
        x3 = self.avg_pool(x3)
        x3 = self.gelu(x3)
        # x3 = self.KanCA_3(x3)
        x3 = self.linear3(x3)

        x4 = self.fc4dws(x4)
        x4 = self.avg_pool(x4)
        x4 = self.gelu(x4)
        # x4 = self.KanCA_4(x4)
        x4 = self.linear4(x4)

        x = (x1 + x2 + x3 + x4) * x
        x = self.sigmoid(x)

        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积层
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, padding, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        # 逐点卷积层
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

