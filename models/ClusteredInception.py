import torch
import torch.nn as nn
import torch.nn.functional as F
from models.inception import BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux
from models.SKNets import SKAttention
from typing import List, Optional, Tuple, Union

class ClusteredInception(nn.Module):
    def __init__(self, configs):
        super(ClusteredInception, self).__init__()
        self.configs = configs
        self.in_channels = configs.enc_in
        
        # 判断是否使用分组模式
        self.use_clustering = hasattr(configs, 'channel_groups') and configs.channel_groups is not None
        
        if self.use_clustering:
            # 分组模式
            self.channel_groups = configs.channel_groups
            self.num_groups = len(self.channel_groups)
            
            # 每个分组的初始卷积层 - 保持图像尺寸不变，添加SKAttention
            self.group_convs = nn.ModuleList([
                nn.Sequential(
                    # 第一层卷积 + SKAttention
                    BasicConv2d(len(group), 32, kernel_size=3, stride=1, padding=1, use_bn=False),
                    SKAttention(channel=32, kernels=[1,3,5], reduction=8),
                    # 第二层卷积 + SKAttention
                    BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1, use_bn=False),
                    SKAttention(channel=32, kernels=[1,3,5], reduction=8),
                    # 第三层卷积 + SKAttention
                    BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1, use_bn=False),
                    SKAttention(channel=48, kernels=[1,3,5], reduction=8),
                    # 1x1卷积 + SKAttention
                    BasicConv2d(48, 48, kernel_size=1, use_bn=False),
                    SKAttention(channel=48, kernels=[1,3], reduction=8),
                ) for group in self.channel_groups
            ])
            
            # 特征融合层 - 将分组特征融合为统一表示，保持空间尺寸，添加SKAttention
            self.fusion_conv = nn.Sequential(
                # 通道融合 + SKAttention
                BasicConv2d(48 * self.num_groups, 192, kernel_size=1, use_bn=False),
                SKAttention(channel=192, kernels=[1,3], reduction=16),
                # 空间特征提取 + SKAttention
                BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1, use_bn=False),
                SKAttention(channel=192, kernels=[1,3,5], reduction=16),
            )
            
            # 合并后的Inception网络
            self.merge_net = InceptionNet(
                num_classes=configs.num_class,
                in_channels=192,  # 融合后的通道数
                aux_logits=False
            )
        else:
            # 直接模式 - 直接使用所有通道
            self.direct_net = InceptionNet(
                num_classes=configs.num_class,
                in_channels=self.in_channels,
                aux_logits=False
            )
        
        # 初始化参数
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入形状: [B, C, H, W]
        batch_size = x.size(0)
        
        if self.use_clustering:
            # 分组处理模式
            group_features = []
            for i, group in enumerate(self.channel_groups):
                # 提取当前分组的通道
                group_channels = x[:, group, :, :]
                # 对当前分组进行卷积，保持空间尺寸，应用SKAttention
                conv_out = self.group_convs[i](group_channels)  # 72x72x48
                group_features.append(conv_out)
            
            # 合并所有分组的特征
            x = torch.cat(group_features, dim=1)  # 72x72x(48*组数)
            # 特征融合，保持空间尺寸，应用SKAttention
            x = self.fusion_conv(x)  # 72x72x192
            # 使用Inception网络进行最终处理
            x = self.merge_net(x)
        else:
            # 直接处理模式
            x = self.direct_net(x)
        
        return x

class InceptionNet(nn.Module):
    """简化版的Inception网络，支持自定义输入通道数，适应小尺寸输入"""
    def __init__(self, num_classes: int, in_channels: int, aux_logits: bool = False):
        super(InceptionNet, self).__init__()
        self.aux_logits = aux_logits
        
        # 修改初始卷积层以适应小尺寸输入
        # 使用更小的stride和padding
        self.Conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)  # 保持尺寸
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 保持尺寸
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 保持尺寸
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 减半
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)  # 1x1卷积不改变尺寸
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, stride=1, padding=1)  # 保持尺寸
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 减半
        
        # 计算经过初始卷积层后的特征图尺寸
        # 假设输入是 72x72，经过两次池化后变为 18x18
        # 调整后续Inception模块的通道数
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 使用自适应池化
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)
        
        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始卷积层
        x = self.Conv2d_1a_3x3(x)  # 72x72
        x = self.Conv2d_2a_3x3(x)  # 72x72
        x = self.Conv2d_2b_3x3(x)  # 72x72
        x = self.maxpool1(x)       # 36x36
        x = self.Conv2d_3b_1x1(x)  # 36x36
        x = self.Conv2d_4a_3x3(x)  # 36x36
        x = self.maxpool2(x)       # 18x18
        
        # Inception模块
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        
        aux = None
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        
        # 分类器
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if self.training and self.aux_logits:
            return x, aux
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        
        # # 如果没有指定通道分组，则使用默认分组
        # if not hasattr(configs, 'channel_groups'):
        #     # 默认将通道平均分组
        #     num_channels = configs.enc_in
        #     channels_per_group = num_channels // 3  # 默认分成3组
        #     configs.channel_groups = [
        #         list(range(i, i + channels_per_group)) 
        #         for i in range(0, num_channels, channels_per_group)
        #     ]
        #     # 处理余数
        #     if num_channels % 3 != 0:
        #         configs.channel_groups[-1].extend(range(num_channels - (num_channels % 3), num_channels))
        
        self.model = ClusteredInception(configs)
        
    def forward(self, x_enc):
        dec_out = self.model(x_enc)
        return dec_out  # [B, N] 