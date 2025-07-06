import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.MultiImageFeatureNet import (
    NoPaddingLargeKernelFeatureExtractor, 
    InceptionFeatureExtractor,
    LargeKernelDilatedFeatureExtractor,
    MultiScaleStackedFeatureExtractor
)


class BasicBlock(nn.Module):
    """ResNet基础块 - 从MultiImageFeatureNet移植并优化"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class GAFOptimizedResNet(nn.Module):
    """专门为GAF图像优化的ResNet - 渐进式下采样保留更多空间信息"""
    def __init__(self, feature_dim=128, depth='resnet18'):
        super().__init__()
        self.depth = depth
        
        if depth == 'resnet18_gaf':
            # GAF优化版本：渐进式下采样，更好保留空间信息
            # 第一层：轻微下采样，保留更多细节
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 96->96，保持尺寸
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            
            # 渐进式下采样的残差层
            self.layer1 = self._make_layer(32, 64, 2, stride=2)    # 96->48
            self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 48->24
            self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 24->12
            self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 12->6
            final_channels = 512
            
        elif depth == 'resnet18_gaf_light':
            # 轻量级GAF版本：适合大数据集
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),  # 96->96
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
            
            self.layer1 = self._make_layer(16, 32, 2, stride=2)    # 96->48
            self.layer2 = self._make_layer(32, 64, 2, stride=2)    # 48->24
            self.layer3 = self._make_layer(64, 128, 2, stride=2)   # 24->12
            self.layer4 = self._make_layer(128, 256, 2, stride=2)  # 12->6
            final_channels = 256
            
        elif depth == 'resnet_gaf_deep':
            # 深度GAF版本：更多层但保持渐进下采样
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 96->96
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 额外的3x3卷积
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            
            self.layer1 = self._make_layer(32, 64, 3, stride=2)    # 96->48
            self.layer2 = self._make_layer(64, 128, 4, stride=2)   # 48->24  
            self.layer3 = self._make_layer(128, 256, 6, stride=2)  # 24->12
            self.layer4 = self._make_layer(256, 512, 3, stride=2)  # 12->6
            final_channels = 512
            
        elif depth == 'resnet_gaf_preserve':
            # 高保真版本：最大程度保留空间信息
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 96->96
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
            self.layer1 = self._make_layer(64, 64, 2, stride=1)    # 96->96，不下采样
            self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 96->48
            self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 48->24
            self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 24->12
            final_channels = 512
            
        else:
            raise ValueError(f"不支持的GAF ResNet深度: {depth}")
        
        # 自适应平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, feature_dim)
        
        print(f"GAF优化ResNet ({depth}) 构建完成:")
        print(f"  - 第一层保持空间分辨率：96x96")
        print(f"  - 渐进式下采样策略")
        print(f"  - 最终通道数：{final_channels}")

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """构建ResNet层"""
        layers = []
        # 第一个块可能需要下采样
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # 后续块不需要下采样
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [N, 1, 96, 96]
        x = self.conv1(x)       # [N, channels, 96, 96] - 保持空间分辨率！
        x = self.layer1(x)      # 根据配置进行下采样
        x = self.layer2(x)      
        x = self.layer3(x)      
        x = self.layer4(x)      
        
        x = self.avgpool(x)     # [N, final_channels, 1, 1]
        x = torch.flatten(x, 1) # [N, final_channels]
        x = self.fc(x)          # [N, feature_dim]
        return x


class ResNetFeatureExtractor(nn.Module):
    """ResNet特征提取器 - 专门为96x96 GAF图像优化"""
    def __init__(self, feature_dim=128, depth='resnet18'):
        super().__init__()
        self.depth = depth
        
        # 新的GAF优化架构
        if depth in ['resnet18_gaf', 'resnet18_gaf_light', 'resnet_gaf_deep', 'resnet_gaf_preserve']:
            self.resnet = GAFOptimizedResNet(feature_dim, depth)
            
        # 保留原有架构以保持兼容性
        elif depth == 'resnet18':
            # 改进的ResNet18：减少初始下采样
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False),  # 96->48，使用5x5而不是7x7
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                # 去掉MaxPool，减少下采样
            )
            
            self.layer1 = self._make_layer(64, 64, 2, stride=1)    # 48->48，不下采样
            self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 48->24
            self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 24->12
            self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 12->6
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, feature_dim)
            
        elif depth == 'resnet34':
            # 改进的ResNet34
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
            self.layer1 = self._make_layer(64, 64, 3, stride=1)    # 48->48
            self.layer2 = self._make_layer(64, 128, 4, stride=2)   # 48->24
            self.layer3 = self._make_layer(128, 256, 6, stride=2)  # 24->12
            self.layer4 = self._make_layer(256, 512, 3, stride=2)  # 12->6
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, feature_dim)
            
        elif depth == 'resnet_light':
            # 轻量级版本，适合DAHU数据集的120个信号
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 96->96，不下采样
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            
            self.layer1 = self._make_layer(32, 64, 2, stride=2)    # 96->48
            self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 48->24
            self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 24->12
            self.layer4 = None  # 去掉最后一层减少参数
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, feature_dim)
            
        else:
            raise ValueError(f"不支持的ResNet深度: {depth}")

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """构建ResNet层"""
        layers = []
        # 第一个块可能需要下采样
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # 后续块不需要下采样
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 如果是新的GAF优化架构，直接使用
        if hasattr(self, 'resnet'):
            return self.resnet(x)
            
        # 原有架构的前向传播
        x = self.conv1(x)       
        x = self.layer1(x)      
        x = self.layer2(x)      
        x = self.layer3(x)      
        
        if self.layer4 is not None:
            x = self.layer4(x)  
        
        x = self.avgpool(x)     
        x = torch.flatten(x, 1) 
        x = self.fc(x)          
        return x


class OptimizedLargeKernelFeatureExtractor(nn.Module):
    """针对96x96优化的大核特征提取器"""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # 96->48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 48->24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 24->12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 12->6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        # x: [N, 1, 96, 96]
        x = self.conv1(x)  # [N, 32, 48, 48]
        x = self.conv2(x)  # [N, 64, 24, 24]
        x = self.conv3(x)  # [N, 128, 12, 12]
        x = self.conv4(x)  # [N, 256, 6, 6]
        x = self.avgpool(x)  # [N, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [N, 256]
        x = self.fc(x)  # [N, feature_dim]
        return x


class OptimizedDilatedFeatureExtractor(nn.Module):
    """针对96x96优化的膨胀卷积特征提取器"""
    def __init__(self, feature_dim=128):
        super().__init__()
        # 初始卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 96->48
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 膨胀卷积层，保持空间分辨率同时增大感受野
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2),  # 保持48x48
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=4, dilation=4),  # 保持48x48
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 最终下采样
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 48->24
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        # x: [N, 1, 96, 96]
        x = self.conv1(x)  # [N, 64, 48, 48]
        x = self.conv2(x)  # [N, 128, 48, 48] (膨胀卷积)
        x = self.conv3(x)  # [N, 256, 48, 48] (膨胀卷积)
        x = self.conv4(x)  # [N, 512, 24, 24]
        x = self.avgpool(x)  # [N, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [N, 512]
        x = self.fc(x)  # [N, feature_dim]
        return x


class AdaptiveFusion(nn.Module):
    """自适应特征融合模块"""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, sum_feat, diff_feat):
        concat_feat = torch.cat([sum_feat, diff_feat], dim=-1)
        weights = self.weight_net(concat_feat)
        return sum_feat * weights + diff_feat * (1 - weights)


class ConcatFusion(nn.Module):
    """简单拼接融合模块"""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.fc = nn.Linear(feature_dim * 2, feature_dim)
    
    def forward(self, sum_feat, diff_feat):
        concat_feat = torch.cat([sum_feat, diff_feat], dim=-1)
        return self.fc(concat_feat)


class ElementwiseFusion(nn.Module):
    """逐元素融合模块"""
    def __init__(self, feature_dim=128, fusion_type='add'):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == 'weighted_add':
            self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, sum_feat, diff_feat):
        if self.fusion_type == 'add':
            return sum_feat + diff_feat
        elif self.fusion_type == 'mul':
            return sum_feat * diff_feat
        elif self.fusion_type == 'weighted_add':
            return self.alpha * sum_feat + (1 - self.alpha) * diff_feat
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")


class BiDirectionalFusion(nn.Module):
    """双向注意力融合模块"""
    def __init__(self, feature_dim=128, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.sum_to_diff = nn.MultiheadAttention(feature_dim, num_heads=num_heads, batch_first=True)
        self.diff_to_sum = nn.MultiheadAttention(feature_dim, num_heads=num_heads, batch_first=True)
        
        # 添加维度变换层：2*feature_dim -> feature_dim
        self.fusion_projection = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, sum_feat, diff_feat):
        # 确保输入是3D张量 [batch_size, seq_len, feature_dim]
        if sum_feat.dim() == 2:
            sum_feat = sum_feat.unsqueeze(1)  # [B, 1, D]
            diff_feat = diff_feat.unsqueeze(1)  # [B, 1, D]
            squeeze_output = True
        else:
            squeeze_output = False
        
        # sum关注diff的信息
        sum_enhanced, _ = self.sum_to_diff(sum_feat, diff_feat, diff_feat)
        # diff关注sum的信息  
        diff_enhanced, _ = self.diff_to_sum(diff_feat, sum_feat, sum_feat)
        
        # 拼接增强后的特征 [B, seq_len, 2*feature_dim]
        concatenated = torch.cat([sum_enhanced, diff_enhanced], dim=-1)
        
        # 通过投影层变换回标准维度 [B, seq_len, feature_dim]
        result = self.fusion_projection(concatenated)
        
        # 如果输入是2D，则将输出也转换为2D
        if squeeze_output:
            result = result.squeeze(1)
        
        return result


class GatedFusion(nn.Module):
    """门控机制融合模块"""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(feature_dim * 2, feature_dim)
        
    def forward(self, sum_feat, diff_feat):
        concat_feat = torch.cat([sum_feat, diff_feat], dim=-1)
        gate_weights = self.gate(concat_feat)
        transformed_feat = self.transform(concat_feat)
        return transformed_feat * gate_weights


class TimeSeriesStatisticalExtractor(nn.Module):
    """时序统计特征提取器
    
    从原始时序数据中提取统计特征和信号间关联特征，
    重点关注信号间的静态关系而非时间依赖性
    """
    def __init__(self, num_signals, time_length, feature_dim=128, stat_type='comprehensive'):
        super().__init__()
        self.num_signals = num_signals
        self.time_length = time_length
        self.feature_dim = feature_dim
        self.stat_type = stat_type
        
        # 计算统计特征的维度
        self.stat_feature_dim = self._calculate_stat_dim()
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(self.stat_feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        print(f"时序统计特征提取器初始化:")
        print(f"  - 信号数量: {num_signals}")
        print(f"  - 时间长度: {time_length}")
        print(f"  - 统计特征维度: {self.stat_feature_dim}")
        print(f"  - 输出特征维度: {feature_dim}")
        print(f"  - 统计类型: {stat_type}")
    
    def _calculate_stat_dim(self):
        """计算统计特征的总维度"""
        if self.stat_type == 'basic':
            # 基础统计：均值、标准差、最大值、最小值、中位数
            return self.num_signals * 5
        elif self.stat_type == 'comprehensive':
            # 综合统计：基础统计 + 相关性矩阵 + 高阶统计
            basic_dim = self.num_signals * 8  # 均值、标准差、最大值、最小值、中位数、偏度、峰度、变异系数
            corr_dim = self.num_signals * (self.num_signals - 1) // 2  # 上三角相关性矩阵
            return basic_dim + corr_dim
        elif self.stat_type == 'correlation_focused':
            # 专注于相关性的统计
            basic_dim = self.num_signals * 5  # 均值、标准差、最大值、最小值、中位数
            corr_dim = self.num_signals * (self.num_signals - 1) // 2  # 相关性矩阵
            cross_dim = self.num_signals * (self.num_signals - 1) // 2 * 2  # 交叉统计特征：每对信号2个特征
            return basic_dim + corr_dim + cross_dim
        else:
            raise ValueError(f"Unknown stat_type: {self.stat_type}")
    
    def extract_statistical_features(self, x):
        """提取统计特征
        
        Args:
            x: [B, T, C] 原始时序数据
            
        Returns:
            features: [B, stat_feature_dim] 统计特征
        """
        B, T, C = x.shape
        features = []
        
        if self.stat_type in ['basic', 'comprehensive', 'correlation_focused']:
            # 基础统计特征（按信号计算）
            mean_vals = torch.mean(x, dim=1)  # [B, C]
            std_vals = torch.std(x, dim=1)    # [B, C]
            max_vals = torch.max(x, dim=1)[0] # [B, C]
            min_vals = torch.min(x, dim=1)[0] # [B, C]
            median_vals = torch.median(x, dim=1)[0]  # [B, C]
            
            features.extend([mean_vals, std_vals, max_vals, min_vals, median_vals])
            
            if self.stat_type == 'comprehensive':
                # 高阶统计特征
                # 偏度（三阶矩）
                centered = x - mean_vals.unsqueeze(1)
                skewness = torch.mean(centered**3, dim=1) / (std_vals**3 + 1e-8)  # [B, C]
                
                # 峰度（四阶矩）
                kurtosis = torch.mean(centered**4, dim=1) / (std_vals**4 + 1e-8)  # [B, C]
                
                # 变异系数
                cv = std_vals / (torch.abs(mean_vals) + 1e-8)  # [B, C]
                
                features.extend([skewness, kurtosis, cv])
        
        # 信号间相关性特征
        if self.stat_type in ['comprehensive', 'correlation_focused']:
            # 计算相关性矩阵
            corr_features = []
            for b in range(B):
                # 计算每个批次的相关性矩阵
                sample = x[b]  # [T, C]
                
                # 标准化
                sample_mean = torch.mean(sample, dim=0, keepdim=True)
                sample_std = torch.std(sample, dim=0, keepdim=True)
                sample_normalized = (sample - sample_mean) / (sample_std + 1e-8)
                
                # 计算相关性矩阵
                corr_matrix = torch.mm(sample_normalized.T, sample_normalized) / (T - 1)
                
                # 提取上三角部分（排除对角线）
                triu_indices = torch.triu_indices(C, C, offset=1)
                corr_values = corr_matrix[triu_indices[0], triu_indices[1]]
                corr_features.append(corr_values)
            
            corr_features = torch.stack(corr_features, dim=0)  # [B, corr_dim]
            features.append(corr_features)
        
        # 交叉统计特征（仅用于correlation_focused）
        if self.stat_type == 'correlation_focused':
            # 信号间的最大差异和最小差异
            signal_diffs = []
            for i in range(C):
                for j in range(i+1, C):
                    diff = x[:, :, i] - x[:, :, j]  # [B, T]
                    max_diff = torch.max(torch.abs(diff), dim=1)[0]  # [B]
                    mean_abs_diff = torch.mean(torch.abs(diff), dim=1)  # [B]
                    signal_diffs.extend([max_diff, mean_abs_diff])
            
            if signal_diffs:
                cross_features = torch.stack(signal_diffs, dim=1)  # [B, cross_dim]
                features.append(cross_features)
        
        # 拼接所有特征
        all_features = torch.cat(features, dim=1)  # [B, stat_feature_dim]
        return all_features
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: [B, T, C] 或 [B, C, T] 原始时序数据
            
        Returns:
            features: [B, feature_dim] 投影后的特征
        """
        # 确保输入格式为 [B, T, C]
        if x.dim() == 3:
            # 优先检查时间维度，然后检查信号维度
            if x.shape[1] == self.time_length and x.shape[2] == self.num_signals:
                # [B, T, C] - 正确格式
                pass
            elif x.shape[1] == self.num_signals and x.shape[2] == self.time_length:
                # [B, C, T] - 需要转置
                x = x.transpose(1, 2)  # [B, T, C]
            else:
                raise ValueError(f"输入张量的形状不匹配：期望形状为[B, {self.time_length}, {self.num_signals}]或[B, {self.num_signals}, {self.time_length}]，但得到形状{x.shape}")
        else:
            raise ValueError(f"输入维度应为3，当前为{x.dim()}")
        
        # 提取统计特征
        stat_features = self.extract_statistical_features(x)  # [B, stat_feature_dim]
        
        # 投影到目标维度
        projected_features = self.feature_projection(stat_features)  # [B, feature_dim]
        
        return projected_features


class MultiModalFusion(nn.Module):
    """多模态融合模块
    
    融合GAF图像特征和时序统计特征
    """
    def __init__(self, feature_dim=128, fusion_strategy='concat'):
        super().__init__()
        self.feature_dim = feature_dim
        self.fusion_strategy = fusion_strategy
        
        if fusion_strategy == 'concat':
            # 简单拼接后投影
            self.fusion_layer = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        elif fusion_strategy == 'attention':
            # 注意力融合
            self.attention = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(feature_dim)
        elif fusion_strategy == 'gated':
            # 门控融合
            self.gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid()
            )
            self.transform = nn.Linear(feature_dim * 2, feature_dim)
        elif fusion_strategy == 'adaptive':
            # 自适应融合
            self.weight_net = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    def forward(self, gaf_features, stat_features):
        """
        Args:
            gaf_features: [B, C, feature_dim] GAF图像特征
            stat_features: [B, feature_dim] 统计特征
            
        Returns:
            fused_features: [B, C, feature_dim] 融合后的特征
        """
        B, C, D = gaf_features.shape
        
        # 将统计特征扩展到每个信号
        stat_features_expanded = stat_features.unsqueeze(1).expand(B, C, D)  # [B, C, feature_dim]
        
        if self.fusion_strategy == 'concat':
            # 拼接融合
            concat_features = torch.cat([gaf_features, stat_features_expanded], dim=-1)  # [B, C, 2*feature_dim]
            fused_features = self.fusion_layer(concat_features)  # [B, C, feature_dim]
            
        elif self.fusion_strategy == 'attention':
            # 注意力融合
            # 将GAF特征作为query，统计特征作为key和value
            gaf_flat = gaf_features.reshape(B*C, 1, D)  # [B*C, 1, D]
            stat_flat = stat_features_expanded.contiguous().view(B*C, 1, D)  # [B*C, 1, D]
            
            attended, _ = self.attention(gaf_flat, stat_flat, stat_flat)  # [B*C, 1, D]
            attended = attended.view(B, C, D)  # [B, C, D]
            
            # 残差连接和层归一化
            fused_features = self.norm(gaf_features + attended)
            
        elif self.fusion_strategy == 'gated':
            # 门控融合
            concat_features = torch.cat([gaf_features, stat_features_expanded], dim=-1)
            gate_weights = self.gate(concat_features)  # [B, C, feature_dim]
            transformed_features = self.transform(concat_features)  # [B, C, feature_dim]
            fused_features = transformed_features * gate_weights
            
        elif self.fusion_strategy == 'adaptive':
            # 自适应融合
            concat_features = torch.cat([gaf_features, stat_features_expanded], dim=-1)
            weights = self.weight_net(concat_features)  # [B, C, feature_dim]
            fused_features = gaf_features * weights + stat_features_expanded * (1 - weights)
        
        return fused_features


class ChannelAttention(nn.Module):
    """通道注意力模块 - 对信号通道维度进行注意力计算"""
    def __init__(self, num_channels, reduction=8):
        super().__init__()
        # 对特征维度进行全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 确保reduction后的通道数至少为1
        reduced_channels = max(1, num_channels // reduction)
        
        # MLP用于学习通道间的关系
        self.fc = nn.Sequential(
            nn.Linear(num_channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, num_channels)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [B, C, feature_dim] 其中C是信号数量
        B, C, D = x.shape
        
        # 对每个信号的特征维度进行全局平均池化和最大池化
        # [B, C, D] -> [B, C, 1] -> [B, C]
        avg_out = self.avg_pool(x).squeeze(-1)  # [B, C]
        max_out = self.max_pool(x).squeeze(-1)  # [B, C]
        
        # 通过MLP学习通道注意力权重
        avg_attention = self.fc(avg_out)  # [B, C]
        max_attention = self.fc(max_out)  # [B, C]
        
        # 融合平均池化和最大池化的注意力
        attention = self.sigmoid(avg_attention + max_attention)  # [B, C]
        
        # 扩展维度并应用注意力权重
        attention = attention.unsqueeze(-1)  # [B, C, 1]
        return x * attention  # [B, C, D]


class SpatialAttention(nn.Module):
    """空间注意力模块（对信号维度）"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [B, C, feature_dim]
        # 对特征维度进行平均和最大操作，得到每个信号的代表性特征
        avg_out = torch.mean(x, dim=2, keepdim=True)  # [B, C, 1]
        max_out, _ = torch.max(x, dim=2, keepdim=True)  # [B, C, 1]
        
        # 拼接平均和最大特征
        x_cat = torch.cat([avg_out, max_out], dim=2)  # [B, C, 2]
        x_cat = x_cat.transpose(1, 2)  # [B, 2, C]
        
        # 通过1D卷积学习信号间的空间关系
        attention = self.conv1(x_cat)  # [B, 1, C]
        attention = attention.transpose(1, 2)  # [B, C, 1]
        attention = self.sigmoid(attention)
        
        return x * attention  # [B, C, feature_dim]


class SignalAttention(nn.Module):
    """信号注意力模块"""
    def __init__(self, feature_dim, num_signals, attention_type='channel'):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == 'channel':
            self.attention = ChannelAttention(num_signals)  # 传入信号数量而不是特征维度
        elif attention_type == 'spatial':
            self.attention = SpatialAttention()
        elif attention_type == 'cbam':
            self.channel_attention = ChannelAttention(num_signals)  # 传入信号数量
            self.spatial_attention = SpatialAttention()
        elif attention_type == 'self':
            self.self_attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    def forward(self, x):
        # x: [B, C, feature_dim]
        if self.attention_type == 'channel':
            return self.attention(x)
        elif self.attention_type == 'spatial':
            return self.attention(x)
        elif self.attention_type == 'cbam':
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
            return x
        elif self.attention_type == 'self':
            # x: [B, C, feature_dim] -> [B, C, feature_dim]
            x_reshaped = x.reshape(x.size(0), x.size(1), -1)
            attn_out, _ = self.self_attention(x_reshaped, x_reshaped, x_reshaped)
            return attn_out


class ResidualBlock(nn.Module):
    """
    残差块，用于分类器中的特征学习
    
    支持不同类型的残差连接：
    - 'basic': 基础残差块
    - 'bottleneck': 瓶颈残差块（适用于高维特征）
    - 'dense': 密集连接残差块
    """
    def __init__(self, in_dim, out_dim, hidden_dim=None, block_type='basic', dropout=0.1):
        super().__init__()
        self.block_type = block_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if hidden_dim is None:
            hidden_dim = max(in_dim, out_dim) // 2
        
        if block_type == 'basic':
            # 基础残差块：输入 -> 线性 -> 激活 -> Dropout -> 线性 -> 输出
            self.main_path = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim)
            )
        elif block_type == 'bottleneck':
            # 瓶颈残差块：降维 -> 处理 -> 升维
            bottleneck_dim = hidden_dim // 2
            self.main_path = nn.Sequential(
                nn.Linear(in_dim, bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim)
            )
        elif block_type == 'dense':
            # 密集连接残差块：更复杂的特征组合
            self.main_path = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim)
            )
        
        # 如果输入输出维度不同，需要投影层
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()
        
        # 最终激活函数
        self.final_activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.main_path(x)
        out = out + residual
        out = self.final_activation(out)
        return out


class ResidualClassifier(nn.Module):
    """
    基于残差块的分类器
    
    Args:
        input_dim (int): 输入特征维度
        num_classes (int): 分类类别数
        hidden_dims (list): 隐藏层维度列表
        block_type (str): 残差块类型
        dropout (float): Dropout比率
        use_batch_norm (bool): 是否使用批归一化
    """
    def __init__(self, input_dim, num_classes, hidden_dims=None, 
                 block_type='basic', dropout=0.1, use_batch_norm=False):
        super().__init__()
        
        if hidden_dims is None:
            # 根据输入维度自动设计网络结构
            if input_dim > 2048:
                hidden_dims = [2048, 1024, 512, 256]
            elif input_dim > 1024:
                hidden_dims = [1024, 512, 256]
            else:
                hidden_dims = [512, 256]
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # 构建残差层序列
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(ResidualBlock(
                in_dim=prev_dim,
                out_dim=hidden_dim,
                block_type=block_type,
                dropout=dropout
            ))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            prev_dim = hidden_dim
        
        # 最终分类层
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
        
        x = self.classifier(x)
        return x


class DualGAFNet(nn.Module):
    """双路GAF网络"""
    def __init__(
        self, 
        feature_dim=32, 
        num_classes=4, 
        num_images=30,
        time_length=96,  # 新增：时间序列长度
        # 特征提取器配置
        extractor_type='large_kernel',  # 'large_kernel', 'inception', 'dilated', 'multiscale'
        # 融合模块配置
        fusion_type='adaptive',  # 'adaptive', 'concat', 'add', 'mul', 'weighted_add'
        # 注意力模块配置
        attention_type='channel',  # 'channel', 'spatial', 'cbam', 'self', 'none'
        # 分类器配置
        classifier_type='mlp',  # 'mlp', 'simple'
        # 统计特征配置
        use_statistical_features=True,  # 是否使用统计特征
        stat_type='comprehensive',  # 'basic', 'comprehensive', 'correlation_focused'
        multimodal_fusion_strategy='concat',  # 'concat', 'attention', 'gated', 'adaptive'
        # 消融实验开关
        use_diff_branch=True,  # 是否使用diff分支进行融合（消融实验）
        # 其他配置
        hvac_groups=None,
        feature_columns=None,
        nhead=4,
        num_layers=2
    ):
        super().__init__()
        self.num_images = num_images
        self.time_length = time_length
        self.feature_dim = feature_dim
        self.extractor_type = extractor_type
        self.fusion_type = fusion_type
        self.attention_type = attention_type
        self.classifier_type = classifier_type
        self.use_statistical_features = use_statistical_features
        self.stat_type = stat_type
        self.multimodal_fusion_strategy = multimodal_fusion_strategy
        # 消融实验开关
        self.use_diff_branch = use_diff_branch
        
        # HVAC分组配置
        self.hvac_groups = hvac_groups
        self.feature_columns = feature_columns if feature_columns else []
        self.use_grouping = hvac_groups is not None
        
        # 智能推荐特征提取器（如果用户没有指定）
        if extractor_type == 'auto':
            recommended_extractor = self._get_recommended_extractor_for_signal_count(num_images)
            print(f"根据信号数量 {num_images} 自动推荐特征提取器: {recommended_extractor}")
            self.extractor_type = recommended_extractor
        
        # 创建特征提取器
        self._build_extractors()
        
        # 创建融合模块（仅在使用diff分支时）
        if self.use_diff_branch:
            self._build_fusion_module()
        
        # 创建统计特征提取器（如果启用）
        if self.use_statistical_features:
            self._build_statistical_extractor()
            self._build_multimodal_fusion()
        
        # 创建信号注意力模块
        self._build_attention_module()
        
        # 创建分类器
        self._build_classifier(num_classes)
        
        # 打印配置信息
        print(f"增强双路GAF网络构建完成:")
        print(f"  - 特征提取器: {self.extractor_type}")
        if self.use_diff_branch:
            print(f"  - 融合模块: {fusion_type}")
        else:
            print(f"  - 融合模块: 已禁用（消融实验：仅使用sum分支）")
        print(f"  - 注意力模块: {attention_type}")
        if attention_type == 'none':
            print(f"    💡 注意力已禁用（消融实验）")
        print(f"  - 分类器: {classifier_type}")
        print(f"  - 信号数量: {num_images}")
        print(f"  - GAF图像尺寸: {time_length}x{time_length}")
        print(f"  - 使用统计特征: {use_statistical_features}")
        if not use_statistical_features:
            print(f"    💡 统计特征已禁用（消融实验）")
        if use_statistical_features:
            print(f"  - 统计特征类型: {stat_type}")
            print(f"  - 多模态融合策略: {multimodal_fusion_strategy}")
        print(f"  - 使用分组: {self.use_grouping}")
        
        # 消融实验状态提示
        ablation_info = []
        if not self.use_diff_branch:
            ablation_info.append("GAF差分分支消融")
        if not self.use_statistical_features:
            ablation_info.append("统计特征消融")
        if self.attention_type == 'none':
            ablation_info.append("注意力机制消融")
        
        if ablation_info:
            print(f"\n🔬 消融实验状态: {' + '.join(ablation_info)}")
        else:
            print(f"\n🔬 消融实验状态: 完整模型（未启用消融）")
        
        # 打印可用的特征提取器类型
        print(f"\n支持的特征提取器类型:")
        print(f"📊 GAF优化ResNet系列 (推荐):")
        print(f"  - resnet18_gaf: GAF优化ResNet18 (适合≤30个信号) ⭐")
        print(f"  - resnet18_gaf_light: 轻量级GAF ResNet (适合30-60个信号) ⭐")
        print(f"  - resnet_gaf_deep: 深层GAF ResNet (更好表达能力)")
        print(f"  - resnet_gaf_preserve: 高保真GAF ResNet (最大程度保留空间信息)")
        print(f"🔧 传统ResNet系列 (兼容性):")
        print(f"  - resnet18: 标准ResNet18 (过早下采样)")
        print(f"  - resnet34: 标准ResNet34 (过早下采样)")
        print(f"  - resnet_light: 轻量级ResNet")
        print(f"⚡ 其他优化架构:")
        print(f"  - optimized_large_kernel: 优化大核卷积 (适合60-120个信号)")
        print(f"  - optimized_dilated: 优化膨胀卷积 (适合>120个信号)")
        print(f"  - large_kernel: 原始大核特征提取器")
        print(f"  - inception: Inception结构特征提取器")
        print(f"  - dilated: 膨胀卷积特征提取器")
        print(f"  - multiscale: 多尺度特征提取器")
        print(f"🤖 智能选择:")
        print(f"  - auto: 根据信号数量自动选择 (推荐GAF优化版本)")

    def _build_extractors(self):
        """构建特征提取器"""
        if self.use_grouping:
            # 使用分组特征提取
            self.channel_to_group = self._create_channel_mapping()
            self.num_groups = len(self.hvac_groups)
            
            # 为每个分支的每个组创建特征提取器
            self.sum_extractors = nn.ModuleDict()
            if self.use_diff_branch:
                self.diff_extractors = nn.ModuleDict()
            
            for group_idx in range(self.num_groups):
                self.sum_extractors[f'group_{group_idx}'] = self._create_extractor()
                if self.use_diff_branch:
                    self.diff_extractors[f'group_{group_idx}'] = self._create_extractor()
            
            # 如果有未分组的通道，创建默认特征提取器
            if -1 in self.channel_to_group.values():
                self.sum_extractors['default'] = self._create_extractor()
                if self.use_diff_branch:
                    self.diff_extractors['default'] = self._create_extractor()
            
            if self.use_diff_branch:
                print(f"创建了 {len(self.sum_extractors)} 个分组特征提取器（sum + diff分支）")
            else:
                print(f"创建了 {len(self.sum_extractors)} 个分组特征提取器（仅sum分支）")
        else:
            # 不使用分组，所有通道共用特征提取器
            self.sum_extractor = self._create_extractor()
            if self.use_diff_branch:
                self.diff_extractor = self._create_extractor()
            
            if self.use_diff_branch:
                print("创建了统一的特征提取器（sum + diff分支）")
            else:
                print("创建了统一的特征提取器（仅sum分支）")

    def _create_extractor(self):
        """根据配置创建特征提取器"""
        if self.extractor_type == 'large_kernel':
            return NoPaddingLargeKernelFeatureExtractor(self.feature_dim)
        elif self.extractor_type == 'inception':
            return InceptionFeatureExtractor(self.feature_dim)
        elif self.extractor_type == 'dilated':
            return LargeKernelDilatedFeatureExtractor(self.feature_dim)
        elif self.extractor_type == 'multiscale':
            return MultiScaleStackedFeatureExtractor(self.feature_dim)
        # 原始ResNet架构（兼容性保留）
        elif self.extractor_type == 'resnet18':
            return ResNetFeatureExtractor(self.feature_dim, depth='resnet18')
        elif self.extractor_type == 'resnet34':
            return ResNetFeatureExtractor(self.feature_dim, depth='resnet34')
        elif self.extractor_type == 'resnet_light':
            return ResNetFeatureExtractor(self.feature_dim, depth='resnet_light')
        # 新的GAF优化ResNet架构
        elif self.extractor_type == 'resnet18_gaf':
            return ResNetFeatureExtractor(self.feature_dim, depth='resnet18_gaf')
        elif self.extractor_type == 'resnet18_gaf_light':
            return ResNetFeatureExtractor(self.feature_dim, depth='resnet18_gaf_light')
        elif self.extractor_type == 'resnet_gaf_deep':
            return ResNetFeatureExtractor(self.feature_dim, depth='resnet_gaf_deep')
        elif self.extractor_type == 'resnet_gaf_preserve':
            return ResNetFeatureExtractor(self.feature_dim, depth='resnet_gaf_preserve')
        # 其他优化的特征提取器
        elif self.extractor_type == 'optimized_large_kernel':
            return OptimizedLargeKernelFeatureExtractor(self.feature_dim)
        elif self.extractor_type == 'optimized_dilated':
            return OptimizedDilatedFeatureExtractor(self.feature_dim)
        else:
            raise ValueError(f"Unknown extractor type: {self.extractor_type}")

    def _get_recommended_extractor_for_signal_count(self, signal_count):
        """根据信号数量推荐特征提取器"""
        if signal_count <= 30:
            return 'resnet18_gaf'  # 小数据集使用GAF优化ResNet18
        elif signal_count <= 60:
            return 'resnet18_gaf_light'  # 中等数据集使用轻量级GAF ResNet
        elif signal_count <= 120:
            return 'optimized_large_kernel'  # 大数据集使用优化的轻量级提取器
        else:
            return 'optimized_dilated'  # 超大数据集使用膨胀卷积减少参数

    def _create_channel_mapping(self):
        """创建通道索引到组的映射"""
        channel_to_group = {}
        
        if not self.feature_columns:
            # 如果没有特征列信息，使用默认映射
            for i in range(self.num_images):
                channel_to_group[i] = 0
            return channel_to_group
        
        # 为每个通道找到对应的组
        for channel_idx, column_name in enumerate(self.feature_columns):
            group_found = False
            
            for group_idx, group_signals in enumerate(self.hvac_groups):
                if any(signal in column_name.upper() for signal in [s.upper() for s in group_signals]):
                    channel_to_group[channel_idx] = group_idx
                    group_found = True
                    break
            
            if not group_found:
                channel_to_group[channel_idx] = -1
        
        return channel_to_group

    def _build_fusion_module(self):
        """构建特征融合模块"""
        if self.fusion_type == 'adaptive':
            self.fusion = AdaptiveFusion(self.feature_dim)
        elif self.fusion_type == 'concat':
            self.fusion = ConcatFusion(self.feature_dim)
        elif self.fusion_type == 'bidirectional':
            self.fusion = BiDirectionalFusion(self.feature_dim, num_heads=4)
        elif self.fusion_type == 'gated':
            self.fusion = GatedFusion(self.feature_dim)
        elif self.fusion_type in ['add', 'mul', 'weighted_add']:
            self.fusion = ElementwiseFusion(self.feature_dim, self.fusion_type)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def _build_statistical_extractor(self):
        """构建统计特征提取器"""
        self.statistical_extractor = TimeSeriesStatisticalExtractor(
            num_signals=self.num_images,
            time_length=self.time_length,
            feature_dim=self.feature_dim,
            stat_type=self.stat_type
        )

    def _build_multimodal_fusion(self):
        """构建多模态融合模块"""
        self.multimodal_fusion = MultiModalFusion(
            feature_dim=self.feature_dim,
            fusion_strategy=self.multimodal_fusion_strategy
        )

    def _build_attention_module(self):
        """构建注意力模块"""
        if self.attention_type == 'none':
            self.attention = nn.Identity()
        else:
            self.attention = SignalAttention(
                self.feature_dim, 
                self.num_images, 
                self.attention_type
            )

    def _build_classifier(self, num_classes):
        """构建分类器"""
        # 所有融合方式现在都输出标准的feature_dim维度
        final_feature_dim = self.feature_dim * self.num_images
        
        if self.classifier_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(final_feature_dim, 2048),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )
        elif self.classifier_type == 'simple':
            self.classifier = nn.Sequential(
                nn.Linear(final_feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
            )
        elif self.classifier_type == 'residual':
            # 基础残差分类器
            self.classifier = ResidualClassifier(
                input_dim=final_feature_dim,
                num_classes=num_classes,
                hidden_dims=[1024, 512, 256],
                block_type='basic',
                dropout=0.1,
                use_batch_norm=False
            )
            print(f"🏗️ 构建残差分类器 (基础残差块): {final_feature_dim} -> [1024, 512, 256] -> {num_classes}")
        elif self.classifier_type == 'residual_bottleneck':
            # 瓶颈残差分类器（适用于高维特征）
            self.classifier = ResidualClassifier(
                input_dim=final_feature_dim,
                num_classes=num_classes,
                hidden_dims=[2048, 1024, 512] if final_feature_dim > 1000 else [1024, 512, 256],
                block_type='bottleneck',
                dropout=0.15,
                use_batch_norm=True
            )
            hidden_info = "[2048, 1024, 512]" if final_feature_dim > 1000 else "[1024, 512, 256]"
            print(f"🏗️ 构建瓶颈残差分类器 (带BatchNorm): {final_feature_dim} -> {hidden_info} -> {num_classes}")
        elif self.classifier_type == 'residual_dense':
            # 密集残差分类器（最强表达能力）
            self.classifier = ResidualClassifier(
                input_dim=final_feature_dim,
                num_classes=num_classes,
                hidden_dims=[1024, 512, 256, 128],
                block_type='dense',
                dropout=0.2,
                use_batch_norm=True
            )
            print(f"🏗️ 构建密集残差分类器 (最强表达能力): {final_feature_dim} -> [1024, 512, 256, 128] -> {num_classes}")
        else:
            raise ValueError(f"不支持的分类器类型: {self.classifier_type}")
        
        print(f"✅ 分类器构建完成: 类型={self.classifier_type}, 输入维度={final_feature_dim}, 输出类别={num_classes}")

    def _extract_features(self, x, extractors_dict, default_extractor):
        """提取特征的通用方法"""
        B, C, H, W = x.shape
        
        if self.use_grouping:
            # 使用分组特征提取
            feats_list = []
            
            # 按组批量处理
            for group_idx in range(self.num_groups):
                group_channels = [ch for ch, g in self.channel_to_group.items() if g == group_idx]
                
                if group_channels:
                    group_x = x[:, group_channels, :, :]
                    group_x = group_x.view(B * len(group_channels), 1, H, W)
                    
                    extractor = extractors_dict[f'group_{group_idx}']
                    group_feats = extractor(group_x)
                    group_feats = group_feats.view(B, len(group_channels), -1)
                    
                    for i, ch in enumerate(group_channels):
                        feats_list.append((ch, group_feats[:, i, :]))
            
            # 处理未分组的通道
            ungrouped_channels = [ch for ch, g in self.channel_to_group.items() if g == -1]
            if ungrouped_channels and 'default' in extractors_dict:
                for ch in ungrouped_channels:
                    channel_x = x[:, ch:ch+1, :, :]
                    channel_feat = extractors_dict['default'](channel_x)
                    feats_list.append((ch, channel_feat))
            
            # 按通道索引排序并堆叠
            feats_list.sort(key=lambda x: x[0])
            feats = torch.stack([feat for _, feat in feats_list], dim=1)
        else:
            # 使用统一特征提取器
            x = x.view(B * C, 1, H, W)
            feats = default_extractor(x)
            feats = feats.view(B, C, -1)
        
        return feats

    def forward(self, sum_x, diff_x, time_series_x=None):
        """前向传播
        
        Args:
            sum_x: [B, C, H, W] Summation GAF图像
            diff_x: [B, C, H, W] Difference GAF图像（消融实验时可忽略）
            time_series_x: [B, C, T] 或 [B, T, C] 原始时序数据（可选）
        """
        B, C, H, W = sum_x.shape
        if self.use_diff_branch:
            assert sum_x.shape == diff_x.shape, "两个分支的输入形状必须相同"
        assert C == self.num_images, f"输入通道数应为{self.num_images}，实际为{C}"
        
        # 如果使用统计特征但没有提供时序数据，抛出错误
        if self.use_statistical_features and time_series_x is None:
            raise ValueError("启用统计特征时必须提供原始时序数据 time_series_x")
        
        # 提取sum分支特征
        if self.use_grouping:
            sum_feats = self._extract_features(sum_x, self.sum_extractors, None)
        else:
            sum_feats = self._extract_features(sum_x, None, self.sum_extractor)
        
        # 根据是否使用diff分支决定特征融合策略
        if self.use_diff_branch:
            # 标准双路GAF：提取diff分支特征并融合
            if self.use_grouping:
                diff_feats = self._extract_features(diff_x, self.diff_extractors, None)
            else:
                diff_feats = self._extract_features(diff_x, None, self.diff_extractor)
            
            # 特征融合
            fused_feats = []
            for i in range(C):
                fused_feat = self.fusion(sum_feats[:, i, :], diff_feats[:, i, :])
                fused_feats.append(fused_feat)
            
            # 所有融合类型现在都输出标准的feature_dim维度
            fused_feats = torch.stack(fused_feats, dim=1)  # [B, C, feature_dim]
        else:
            # 消融实验：仅使用sum分支，不进行融合
            fused_feats = sum_feats  # [B, C, feature_dim]
        
        # 如果使用统计特征，进行多模态融合
        if self.use_statistical_features:
            # 提取统计特征
            stat_features = self.statistical_extractor(time_series_x)  # [B, feature_dim]
            
            # 多模态融合
            fused_feats = self.multimodal_fusion(fused_feats, stat_features)  # [B, C, feature_dim]
        
        # 信号注意力（如果attention_type='none'，则相当于Identity）
        attended_feats = self.attention(fused_feats)  # [B, C, feature_dim]
        
        # 展平用于分类
        merged = attended_feats.reshape(B, -1)  # [B, C * feature_dim]
        
        # 分类
        out = self.classifier(merged)
        return out


class Model(nn.Module):
    """双路GAF网络的包装类"""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        
        # 从配置中获取参数
        feature_dim = configs.feature_dim
        num_classes = configs.num_class
        num_images = configs.enc_in
        time_length = getattr(configs, 'seq_len', 96)  # 时间序列长度
        
        # 获取模块配置
        extractor_type = getattr(configs, 'extractor_type', 'large_kernel')
        fusion_type = getattr(configs, 'fusion_type', 'adaptive')
        attention_type = getattr(configs, 'attention_type', 'channel')
        classifier_type = getattr(configs, 'classifier_type', 'mlp')
        
        # 获取统计特征配置
        use_statistical_features = getattr(configs, 'use_statistical_features', True)
        stat_type = getattr(configs, 'stat_type', 'comprehensive')
        multimodal_fusion_strategy = getattr(configs, 'multimodal_fusion_strategy', 'concat')
        
        # 获取消融实验配置
        use_diff_branch = getattr(configs, 'use_diff_branch', True)
        
        # 获取HVAC信号分组配置
        hvac_groups = getattr(configs, 'hvac_groups', None)
        feature_columns = getattr(configs, 'feature_columns', None)
        
        self.model = DualGAFNet(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_images=num_images,
            time_length=time_length,
            extractor_type=extractor_type,
            fusion_type=fusion_type,
            attention_type=attention_type,
            classifier_type=classifier_type,
            use_statistical_features=use_statistical_features,
            stat_type=stat_type,
            multimodal_fusion_strategy=multimodal_fusion_strategy,
            use_diff_branch=use_diff_branch,
            hvac_groups=hvac_groups,
            feature_columns=feature_columns
        )

    def forward(self, sum_x, diff_x, time_series_x=None):
        return self.model(sum_x, diff_x, time_series_x)


if __name__ == "__main__":
    print("="*100)
    print("测试双路GAF网络 - 针对96x96 GAF图像优化")
    print("="*100)
    
    # 测试不同信号数量和特征提取器
    test_configs = [
        (26, 96, "resnet18", "SAHU数据集"),
        (120, 96, "auto", "DAHU数据集（自动选择）"),
        (120, 96, "resnet_light", "DAHU数据集（手动轻量级）"),
        (120, 96, "optimized_large_kernel", "DAHU数据集（优化大核）"),
    ]
    
    for signal_count, gaf_size, extractor_type, dataset_name in test_configs:
        print(f"\n{'-'*60}")
        print(f"测试配置: {dataset_name}")
        print(f"信号数量: {signal_count}, GAF尺寸: {gaf_size}x{gaf_size}")
        print(f"特征提取器: {extractor_type}")
        print(f"{'-'*60}")
        
        # 创建测试数据
        B = 2  # 小批次测试
        C, H, W = signal_count, gaf_size, gaf_size
        sum_x = torch.randn(B, C, H, W)
        diff_x = torch.randn(B, C, H, W)
        time_series_x = torch.randn(B, gaf_size, C)  # 时序数据
        
        # 创建配置
        configs = type("cfg", (), {
            "feature_dim": 64,  # 使用较小的特征维度进行测试
            "num_class": 6,     # DAHU数据集有更多类别
            "enc_in": C,
            "seq_len": gaf_size,
            "extractor_type": extractor_type,
            "fusion_type": "adaptive", 
            "attention_type": "channel",
            "classifier_type": "mlp",
            "use_statistical_features": True,
            "stat_type": "basic",  # 使用基础统计特征减少计算
            "multimodal_fusion_strategy": "concat",
            "hvac_groups": None,
            "feature_columns": None
        })()
        
        try:
            model = Model(configs)
            
            print(f"\n输入数据:")
            print(f"  - Summation GAF: {sum_x.shape}")
            print(f"  - Difference GAF: {diff_x.shape}")
            print(f"  - Time Series: {time_series_x.shape}")
            
            # 测试forward过程
            out = model(sum_x, diff_x, time_series_x)
            print(f"\n模型输出:")
            print(f"  - 输出 shape: {out.shape}")
            print(f"  - 输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
            
            # 计算参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n模型复杂度:")
            print(f"  - 总参数数量: {total_params:,}")
            print(f"  - 可训练参数: {trainable_params:,}")
            print(f"  - 模型大小估计: {total_params * 4 / (1024**2):.2f} MB")
            
            # 计算内存估计
            input_memory = (sum_x.numel() + diff_x.numel() + time_series_x.numel()) * 4 / (1024**2)
            print(f"  - 单batch输入内存: {input_memory:.2f} MB")
            
            print(f"✅ 测试通过")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            print(f"错误类型: {type(e).__name__}")
    
    print(f"\n{'='*100}")
    print("测试完成")
    print("="*100) 