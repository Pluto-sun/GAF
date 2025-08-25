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
from models.ChannelCompressionModule import (
    SignalCompressionModule,
    ChannelCompressionModule,
    AdaptiveChannelCompressionModule,
    HVACSignalGroupCompressionModule
)


class BasicBlock(nn.Module):
    """ResNet基础块 - 从MultiImageFeatureNet移植并优化"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride, 1, bias=False)
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
        # out += identity
        out = self.relu(out)
        return out
class FiLMFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        # 将统计特征映射为 γ 和 β
        self.gamma_net = nn.Linear(feature_dim, feature_dim)
        self.beta_net = nn.Linear(feature_dim, feature_dim)

        # 初始化：让初始状态接近恒等变换
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.constant_(self.beta_net.bias, 0.0)

    def forward(self, gaf_features, stat_features):
        B, C, D = gaf_features.shape

        # ✅ 强烈建议：对统计特征归一化
        stat_normalized = F.normalize(stat_features, p=2, dim=1)  # [B, D]

        # 生成 γ 和 β
        gamma = 1 + self.gamma_net(stat_normalized)  # 初始为 1
        beta = self.beta_net(stat_normalized)        # 初始为 0

        # 扩展到每个通道
        gamma = gamma.unsqueeze(1)  # [B, 1, D]
        beta = beta.unsqueeze(1)    # [B, 1, D]

        # ✅ gaf_features 是否归一化？可选
        # 方案A：保留原始强度
        fused_features = gamma * gaf_features + beta

        # 方案B：先归一化（更稳定）
        # gaf_norm = F.normalize(gaf_features, p=2, dim=-1)
        # fused_features = gamma * gaf_norm + beta

        return fused_features

class GAFOptimizedResNet(nn.Module):
    """专门为GAF图像优化的ResNet - 渐进式下采样保留更多空间信息"""

    def __init__(self, feature_dim=128, depth='resnet18'):
        super().__init__()
        self.depth = depth

        if depth == 'resnet18_gaf':
            # GAF优化版本：渐进式下采样，更好保留空间信息
            # 第一层：轻微下采样，保留更多细节
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 96->96，保持尺寸
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
                nn.Conv2d(1, 16, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 96->96
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
                nn.Conv2d(1, 32, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 96->96
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 额外的3x3卷积
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
                nn.Conv2d(1, 64, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 96->96
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_layer(
                64, 64, 2, stride=1)    # 96->96，不下采样
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
        x = torch.flatten(x, 1)  # [N, final_channels]
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
                nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2,
                          bias=False),  # 96->48，使用5x5而不是7x7
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                # 去掉MaxPool，减少下采样
            )

            self.layer1 = self._make_layer(
                64, 64, 2, stride=1)    # 48->48，不下采样
            self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 48->24
            self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 24->12
            self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 12->6

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, feature_dim)

        elif depth == 'resnet34':
            # 改进的ResNet34
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=5, stride=2,
                          padding=2, bias=False),
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
                nn.Conv2d(1, 32, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 96->96，不下采样
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
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # 96->48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 膨胀卷积层，保持空间分辨率同时增大感受野
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1,
                      padding=2, dilation=2),  # 保持48x48
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1,
                      padding=4, dilation=4),  # 保持48x48
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 最终下采样
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 48->24
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        # x: [N, 1, 96, 96]
        x = self.conv1(x)  # [N, 32, 48, 48]
        x = self.conv2(x)  # [N, 64, 24, 24] (膨胀卷积)
        x = self.conv3(x)  # [N, 128, 12, 12] (膨胀卷积)
        x = self.conv4(x)  # [N, 256, 6, 6]
        x = self.avgpool(x)  # [N, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [N, 256]
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


# class BiDirectionalFusion(nn.Module):
#     """双向注意力融合模块"""

#     def __init__(self, feature_dim=128, num_heads=4):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.sum_to_diff = nn.MultiheadAttention(
#             feature_dim, num_heads=num_heads, batch_first=True)
#         self.diff_to_sum = nn.MultiheadAttention(
#             feature_dim, num_heads=num_heads, batch_first=True)

#         # 添加维度变换层：2*feature_dim -> feature_dim
#         self.fusion_projection = nn.Sequential(
#             nn.Linear(2 * feature_dim, feature_dim),
#             nn.LayerNorm(feature_dim),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, sum_feat, diff_feat):
#         # 确保输入是3D张量 [batch_size, seq_len, feature_dim]
#         if sum_feat.dim() == 2:
#             sum_feat = sum_feat.unsqueeze(1)  # [B, 1, D]
#             diff_feat = diff_feat.unsqueeze(1)  # [B, 1, D]
#             squeeze_output = True
#         else:
#             squeeze_output = False

#         # sum关注diff的信息
#         sum_enhanced, _ = self.sum_to_diff(sum_feat, diff_feat, diff_feat)
#         # diff关注sum的信息
#         diff_enhanced, _ = self.diff_to_sum(diff_feat, sum_feat, sum_feat)

#         # 拼接增强后的特征 [B, seq_len, 2*feature_dim]
#         concatenated = torch.cat([sum_enhanced, diff_enhanced], dim=-1)

#         # 通过投影层变换回标准维度 [B, seq_len, feature_dim]
#         result = self.fusion_projection(concatenated)

#         # 如果输入是2D，则将输出也转换为2D
#         if squeeze_output:
#             result = result.squeeze(1)

#         return result
class BiDirectionalFusion(nn.Module):
    """双向注意力融合模块：融合 sum 和 diff 类型 GAF 向量"""

    def __init__(self, feature_dim=128, num_heads=4, use_ffn=True, ffn_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_ffn = use_ffn

        # 双向交叉注意力
        self.sum_to_diff = nn.MultiheadAttention(
            feature_dim, num_heads=num_heads, batch_first=True
        )
        self.diff_to_sum = nn.MultiheadAttention(
            feature_dim, num_heads=num_heads, batch_first=True
        )

        # LayerNorm for each attention output
        self.sum_norm = nn.LayerNorm(feature_dim)
        self.diff_norm = nn.LayerNorm(feature_dim)

        # 拼接后的特征投影层
        self.fusion_projection = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )

        # 可选：前馈网络（FFN）增强表达能力
        if self.use_ffn:
            ffn_inner_dim = ffn_dim if ffn_dim is not None else 4 * feature_dim
            self.ffn = nn.Sequential(
                nn.Linear(feature_dim, ffn_inner_dim),
                nn.GELU(),
                nn.Linear(ffn_inner_dim, feature_dim)
            )
            self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self, sum_feat, diff_feat):
        # 确保输入是3D张量 [batch_size, seq_len, feature_dim]
        if sum_feat.dim() == 2:
            sum_feat = sum_feat.unsqueeze(1)  # [B, 1, D]
            diff_feat = diff_feat.unsqueeze(1)  # [B, 1, D]
            squeeze_output = True
        else:
            squeeze_output = False

        # sum 特征关注 diff 特征
        sum_enhanced, _ = self.sum_to_diff(sum_feat, diff_feat, diff_feat)
        sum_enhanced = self.sum_norm(sum_enhanced + sum_feat)  # 残差 + LayerNorm

        # diff 特征关注 sum 特征
        diff_enhanced, _ = self.diff_to_sum(diff_feat, sum_feat, sum_feat)
        diff_enhanced = self.diff_norm(
            diff_enhanced + diff_feat)  # 残差 + LayerNorm

        # 拼接增强后的特征
        concatenated = torch.cat(
            [sum_enhanced, diff_enhanced], dim=-1)  # [B, C, 2D]

        # 投影回原始维度
        fused = self.fusion_projection(concatenated)  # [B, C, D]

        # 可选：前馈网络进一步增强表达
        if self.use_ffn:
            fused = fused + self.ffn(fused)  # 残差连接
            fused = self.ffn_norm(fused)  # LayerNorm

        # 如果输入是2D，则将输出也转换为2D
        if squeeze_output:
            fused = fused.squeeze(1)

        return fused


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

        # 统计特征维度
        if stat_type == 'basic':
            # 5(均值/方差/最大/最小/中位数) + 2(偏度/峰度) + 4(10/25/75/90分位数) + 1(iqr) + 2(变化率均值/方差) = 14
            self.basic_dim = num_signals * 5
        elif stat_type in ['comprehensive', 'correlation_focused']:
            self.basic_dim = num_signals * 5
        else:
            self.basic_dim = num_signals * 5
        self.corr_dim = num_signals * (num_signals - 1) // 2 if stat_type in ['comprehensive', 'correlation_focused'] else 0
        self.diff_dim = num_signals * (num_signals - 1) if stat_type == 'correlation_focused' else 0

        # 降维 MLP
        if stat_type in ['comprehensive', 'correlation_focused']:
            self.basic_proj = nn.Sequential(
                nn.Linear(self.basic_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.LayerNorm(128)
            )
            self.corr_proj = nn.Sequential(
                nn.Linear(self.corr_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.LayerNorm(128)
            )
            if stat_type == 'correlation_focused':
                self.diff_proj = nn.Sequential(
                    nn.Linear(self.diff_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 128),
                    nn.LayerNorm(128)
                )
            else:
                self.diff_proj = None
            final_in_dim = 128 + 128 + (128 if self.diff_proj is not None else 0)
            self.final_proj = nn.Sequential(
                nn.Linear(final_in_dim, feature_dim*2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim*2, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        else:
            self.feature_projection = nn.Sequential(
                nn.Linear(self.basic_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, feature_dim),
                nn.LayerNorm(feature_dim)
            )

        print(f"时序统计特征提取器初始化:")
        print(f"  - 信号数量: {num_signals}")
        print(f"  - 时间长度: {time_length}")
        print(f"  - 统计特征维度: {self.basic_dim + self.corr_dim + self.diff_dim}")
        print(f"  - 输出特征维度: {feature_dim}")
        print(f"  - 统计类型: {stat_type}")

    def _calculate_stat_dim(self):
        if self.stat_type == 'basic':
            return self.num_signals * 14
        elif self.stat_type == 'comprehensive':
            basic_dim = self.num_signals * 5
            corr_dim = self.num_signals * (self.num_signals - 1) // 2
            return basic_dim + corr_dim
        elif self.stat_type == 'correlation_focused':
            basic_dim = self.num_signals * 5
            corr_dim = self.num_signals * (self.num_signals - 1) // 2
            cross_dim = self.num_signals * (self.num_signals - 1)
            return basic_dim + corr_dim + cross_dim
        else:
            raise ValueError(f"Unknown stat_type: {self.stat_type}")

    def extract_statistical_features(self, x):
        B, T, C = x.shape
        # 基础统计
        if self.stat_type == 'basic':
            mean_vals = torch.mean(x, dim=1)  # [B, C]
            std_vals = torch.std(x, dim=1)    # [B, C]
            max_vals = torch.max(x, dim=1)[0]  # [B, C]
            min_vals = torch.min(x, dim=1)[0]  # [B, C]
            median_vals = torch.median(x, dim=1)[0]  # [B, C]

            # 偏度、峰度
            centered = x - mean_vals.unsqueeze(1)
            skewness = torch.mean(centered**3, dim=1) / (std_vals**3 + 1e-8)  # [B, C]
            kurtosis = torch.mean(centered**4, dim=1) / (std_vals**4 + 1e-8)  # [B, C]

            # 分位数
            q10 = torch.quantile(x, 0.10, dim=1)  # [B, C]
            q25 = torch.quantile(x, 0.25, dim=1)  # [B, C]
            q75 = torch.quantile(x, 0.75, dim=1)  # [B, C]
            q90 = torch.quantile(x, 0.90, dim=1)  # [B, C]
            iqr = q75 - q25  # [B, C]
            # 变化率
            diff = x[:, 1:, :] - x[:, :-1, :]  # [B, T-1, C]
            mean_diff_rate = torch.mean(diff, dim=1)  # [B, C]
            std_diff_rate = torch.std(diff, dim=1)    # [B, C]
            # 拼接所有特征
            # feats = [mean_vals, std_vals, max_vals, min_vals, median_vals,
            #          skewness, kurtosis, q10, q25, q75, q90, iqr, mean_diff_rate, std_diff_rate]
            feats = [mean_vals, std_vals, max_vals, min_vals, median_vals]
            basic_feats = torch.cat(feats, dim=1)  # [B, C*14]
            return basic_feats, None, None
        else:
            # 其它模式保持原有逻辑
            mean_vals = torch.mean(x, dim=1)  # [B, C]
            std_vals = torch.std(x, dim=1)    # [B, C]
            max_vals = torch.max(x, dim=1)[0]  # [B, C]
            min_vals = torch.min(x, dim=1)[0]  # [B, C]
            median_vals = torch.median(x, dim=1)[0]  # [B, C]
            basic_feats = [mean_vals, std_vals, max_vals, min_vals, median_vals]
            basic_feats = torch.cat(basic_feats, dim=1)  # [B, basic_dim]
            # 相关性
            if self.stat_type in ['comprehensive', 'correlation_focused']:
                corr_features = []
                for b in range(B):
                    sample = x[b]
                    sample_mean = torch.mean(sample, dim=0, keepdim=True)
                    sample_std = torch.std(sample, dim=0, keepdim=True)
                    sample_normalized = (sample - sample_mean) / (sample_std + 1e-8)
                    corr_matrix = torch.mm(sample_normalized.T, sample_normalized) / (T - 1)
                    triu_indices = torch.triu_indices(C, C, offset=1)
                    corr_values = corr_matrix[triu_indices[0], triu_indices[1]]
                    corr_features.append(corr_values)
                corr_features = torch.stack(corr_features, dim=0)  # [B, corr_dim]
            else:
                corr_features = None
            # 差异特征
            if self.stat_type == 'correlation_focused':
                signal_diffs = []
                for i in range(C):
                    for j in range(i+1, C):
                        diff = x[:, :, i] - x[:, :, j]  # [B, T]
                        max_diff = torch.max(torch.abs(diff), dim=1)[0]  # [B]
                        mean_abs_diff = torch.mean(torch.abs(diff), dim=1)  # [B]
                        signal_diffs.extend([max_diff, mean_abs_diff])
                signal_diffs = torch.stack(signal_diffs, dim=1)  # [B, diff_dim]
            else:
                signal_diffs = None
            return basic_feats, corr_features, signal_diffs

    def forward(self, x):
        """前向传播
        Args:
            x: [B, T, C] 或 [B, C, T] 原始时序数据
        Returns:
            features: [B, feature_dim] 投影后的特征
        """
        if x.dim() == 3:
            if x.shape[1] == self.time_length and x.shape[2] == self.num_signals:
                pass
            elif x.shape[1] == self.num_signals and x.shape[2] == self.time_length:
                x = x.transpose(1, 2)
            else:
                raise ValueError(f"输入张量的形状不匹配：期望形状为[B, {self.time_length}, {self.num_signals}]或[B, {self.num_signals}, {self.time_length}]，但得到形状{x.shape}")
        else:
            raise ValueError(f"输入维度应为3，当前为{x.dim()}")
        if self.stat_type in ['comprehensive', 'correlation_focused']:
            basic_feats, corr_features, signal_diffs = self.extract_statistical_features(x)
            basic_emb = self.basic_proj(basic_feats)
            if corr_features is not None:
                corr_emb = self.corr_proj(corr_features)
            else:
                corr_emb = torch.zeros(basic_emb.shape[0], 128, device=basic_emb.device, dtype=basic_emb.dtype)
            if self.stat_type == 'correlation_focused':
                if signal_diffs is not None:
                    diff_emb = self.diff_proj(signal_diffs)
                else:
                    diff_emb = torch.zeros(basic_emb.shape[0], 128, device=basic_emb.device, dtype=basic_emb.dtype)
                all_emb = torch.cat([basic_emb, corr_emb, diff_emb], dim=1)
            else:
                all_emb = torch.cat([basic_emb, corr_emb], dim=1)
            projected_features = self.final_proj(all_emb)
        else:
            stat_features = self.extract_statistical_features(x)[0]
            projected_features = self.feature_projection(stat_features)
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
            self.attention = nn.MultiheadAttention(
                feature_dim, num_heads=4, batch_first=True)
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
        elif fusion_strategy == 'film':
            # 将统计特征映射为 γ 和 β
            self.gamma_net = nn.Linear(feature_dim, feature_dim)
            self.beta_net = nn.Linear(feature_dim, feature_dim)

            # 初始化：让初始状态接近恒等变换
            nn.init.zeros_(self.gamma_net.bias)
            nn.init.constant_(self.beta_net.bias, 0.0)
            # self.film = FiLMFusion(feature_dim)
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
        gaf_normalized = F.normalize(gaf_features, p=2, dim=2)  # [B, C, D]
        stat_normalized = F.normalize(stat_features, p=2, dim=1)  # [B, D]
        # 将统计特征扩展到每个信号
        stat_features_expanded = stat_features.unsqueeze(
            1).expand(B, C, D)  # [B, C, feature_dim]
        stat_expanded = stat_normalized.unsqueeze(1).expand_as(gaf_normalized)  # [B, C, D]
        if self.fusion_strategy == 'concat':
            # 拼接融合
            concat_features = torch.cat(
                [gaf_normalized, stat_expanded], dim=-1)  # [B, C, 2*feature_dim]
            fused_features = self.fusion_layer(
                concat_features)  # [B, C, feature_dim]

        elif self.fusion_strategy == 'attention':
            # 注意力融合
            # 将GAF特征作为query，统计特征作为key和value
            gaf_flat = gaf_features.reshape(B*C, 1, D)  # [B*C, 1, D]
            stat_flat = stat_features_expanded.contiguous().view(B*C, 1,
                                                                 D)  # [B*C, 1, D]

            attended, _ = self.attention(
                gaf_flat, stat_flat, stat_flat)  # [B*C, 1, D]
            attended = attended.view(B, C, D)  # [B, C, D]

            # 残差连接和层归一化
            fused_features = self.norm(gaf_features + attended)

        elif self.fusion_strategy == 'gated':
            # 门控融合
            concat_features = torch.cat(
                [gaf_normalized, stat_expanded], dim=-1)
            gate_weights = self.gate(concat_features)  # [B, C, feature_dim]
            # transformed_features = self.transform(
            #     concat_features)  # [B, C, feature_dim]
            fused_features = gate_weights * gaf_normalized + (1 - gate_weights) * stat_expanded
            # fused_features = transformed_features * gate_weights

        elif self.fusion_strategy == 'adaptive':
            # 自适应融合
            concat_features = torch.cat(
                [gaf_normalized, stat_expanded], dim=-1)
            weights = self.weight_net(concat_features)  # [B, C, feature_dim]
            fused_features = gaf_features * weights + \
                stat_features_expanded * (1 - weights)
        elif self.fusion_strategy == 'film':
            # 生成 γ 和 β
            gamma = 1 + self.gamma_net(stat_normalized)  # 初始为 1
            beta = self.beta_net(stat_normalized)        # 初始为 0

            # 扩展到每个通道
            gamma = gamma.unsqueeze(1)  # [B, 1, D]
            beta = beta.unsqueeze(1)    # [B, 1, D]

            # ✅ gaf_features 是否归一化？可选
            # 方案A：保留原始强度
            fused_features = gamma * gaf_features + beta
        return fused_features


class SignalLevelStatisticalExtractor(nn.Module):
    """信号级统计特征提取器

    为每个信号单独提取统计特征，提供更好的可解释性和信号特定性
    """

    def __init__(self, num_signals, time_length, stat_feature_dim=32, stat_type='comprehensive'):
        super().__init__()
        self.num_signals = num_signals
        self.time_length = time_length
        self.stat_feature_dim = stat_feature_dim
        self.stat_type = stat_type

        # 计算每个信号的统计特征数量
        self.single_signal_stat_dim = self._calculate_single_signal_stat_dim()

        # 为每个信号的统计特征创建投影层
        self.signal_stat_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.single_signal_stat_dim, stat_feature_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(stat_feature_dim * 2, stat_feature_dim),
                nn.LayerNorm(stat_feature_dim)
            ) for _ in range(num_signals)
        ])

        print(f"信号级统计特征提取器初始化:")
        print(f"  - 信号数量: {num_signals}")
        print(f"  - 时间长度: {time_length}")
        print(f"  - 单信号统计特征维度: {self.single_signal_stat_dim}")
        print(f"  - 输出特征维度: {stat_feature_dim}")
        print(f"  - 统计类型: {stat_type}")

    def _calculate_single_signal_stat_dim(self):
        """计算单个信号的统计特征维度"""
        if self.stat_type == 'basic':
            # 基础统计：均值、标准差、最大值、最小值、中位数
            return 5
        elif self.stat_type == 'comprehensive':
            # 综合统计：均值、标准差、最大值、最小值、中位数、偏度、峰度、变异系数
            return 8
        elif self.stat_type == 'extended':
            # 扩展统计：基础统计(5) + 高阶统计(3) + 扩展统计(4) = 12
            # 基础：mean, std, max, min, median
            # 高阶：skewness, kurtosis, cv
            # 扩展：percentile_25, percentile_75, range_val, mad
            return 12
        else:
            raise ValueError(f"Unknown stat_type: {self.stat_type}")

    def extract_signal_level_features(self, x):
        """提取信号级统计特征

        Args:
            x: [B, T, C] 原始时序数据

        Returns:
            features: [B, C, single_signal_stat_dim] 每个信号的统计特征
        """
        B, T, C = x.shape
        signal_features = []

        for c in range(C):
            signal_data = x[:, :, c]  # [B, T] 单个信号的数据

            # 基础统计特征
            mean_val = torch.mean(signal_data, dim=1)  # [B]
            std_val = torch.std(signal_data, dim=1)    # [B]
            max_val = torch.max(signal_data, dim=1)[0]  # [B]
            min_val = torch.min(signal_data, dim=1)[0]  # [B]
            median_val = torch.median(signal_data, dim=1)[0]  # [B]

            features = [mean_val, std_val, max_val, min_val, median_val]

            if self.stat_type in ['comprehensive', 'extended']:
                # 高阶统计特征
                centered = signal_data - mean_val.unsqueeze(1)  # [B, T]
                skewness = torch.mean(centered**3, dim=1) / \
                    (std_val**3 + 1e-8)  # [B]
                kurtosis = torch.mean(centered**4, dim=1) / \
                    (std_val**4 + 1e-8)  # [B]
                cv = std_val / (torch.abs(mean_val) + 1e-8)  # [B] 变异系数

                features.extend([skewness, kurtosis, cv])

            if self.stat_type == 'extended':
                # 扩展统计特征
                # 25% 和 75% 百分位数
                percentile_25 = torch.quantile(signal_data, 0.25, dim=1)  # [B]
                percentile_75 = torch.quantile(signal_data, 0.75, dim=1)  # [B]

                # 范围和四分位距
                range_val = max_val - min_val  # [B]

                # 均值绝对偏差
                mad = torch.mean(
                    torch.abs(signal_data - mean_val.unsqueeze(1)), dim=1)  # [B]

                features.extend([percentile_25, percentile_75, range_val, mad])

            # 堆叠单个信号的所有特征
            # [B, single_signal_stat_dim]
            signal_feat = torch.stack(features, dim=1)
            signal_features.append(signal_feat)

        # 堆叠所有信号的特征
        # [B, C, single_signal_stat_dim]
        all_signal_features = torch.stack(signal_features, dim=1)
        return all_signal_features

    def forward(self, x):
        """前向传播

        Args:
            x: [B, T, C] 或 [B, C, T] 原始时序数据

        Returns:
            features: [B, C, stat_feature_dim] 每个信号的投影统计特征
        """
        # 确保输入格式为 [B, T, C]
        if x.dim() == 3:
            if x.shape[1] == self.time_length and x.shape[2] == self.num_signals:
                # [B, T, C] - 正确格式
                pass
            elif x.shape[1] == self.num_signals and x.shape[2] == self.time_length:
                # [B, C, T] - 需要转置
                x = x.transpose(1, 2)  # [B, T, C]
            else:
                raise ValueError(
                    f"输入张量的形状不匹配：期望形状为[B, {self.time_length}, {self.num_signals}]或[B, {self.num_signals}, {self.time_length}]，但得到形状{x.shape}")
        else:
            raise ValueError(f"输入维度应为3，当前为{x.dim()}")

        # 提取信号级统计特征
        signal_stat_features = self.extract_signal_level_features(
            x)  # [B, C, single_signal_stat_dim]

        # 为每个信号应用对应的投影层
        projected_features = []
        for c in range(self.num_signals):
            # [B, single_signal_stat_dim]
            signal_feat = signal_stat_features[:, c, :]
            projected_feat = self.signal_stat_projections[c](
                signal_feat)  # [B, stat_feature_dim]
            projected_features.append(projected_feat)

        # 堆叠所有信号的投影特征
        all_projected_features = torch.stack(
            projected_features, dim=1)  # [B, C, stat_feature_dim]

        return all_projected_features


class SignalLevelStatisticalFusion(nn.Module):
    """信号级统计特征融合模块

    将GAF特征和信号级统计特征进行融合，支持多种融合策略
    """

    def __init__(self, gaf_feature_dim, stat_feature_dim, output_feature_dim=None,
                 fusion_strategy='concat_project'):
        super().__init__()
        self.gaf_feature_dim = gaf_feature_dim
        self.stat_feature_dim = stat_feature_dim
        self.output_feature_dim = output_feature_dim or gaf_feature_dim
        self.fusion_strategy = fusion_strategy

        self._build_fusion_module()

        print(f"信号级统计特征融合模块初始化:")
        print(f"  - GAF特征维度: {gaf_feature_dim}")
        print(f"  - 统计特征维度: {stat_feature_dim}")
        print(f"  - 输出特征维度: {self.output_feature_dim}")
        print(f"  - 融合策略: {fusion_strategy}")

    def _build_fusion_module(self):
        """构建融合模块"""
        if self.fusion_strategy == 'concat_project':
            # 策略1：直接拼接后投影
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.gaf_feature_dim + self.stat_feature_dim,
                          self.output_feature_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.output_feature_dim *
                          2, self.output_feature_dim),
                nn.LayerNorm(self.output_feature_dim)
            )

        elif self.fusion_strategy == 'attention_fusion':
            # 策略2：注意力融合
            self.gaf_attention = nn.Sequential(
                nn.Linear(self.gaf_feature_dim +
                          self.stat_feature_dim, self.gaf_feature_dim),
                nn.Sigmoid()
            )
            self.stat_attention = nn.Sequential(
                nn.Linear(self.gaf_feature_dim +
                          self.stat_feature_dim, self.stat_feature_dim),
                nn.Sigmoid()
            )
            self.output_projection = nn.Linear(
                self.gaf_feature_dim + self.stat_feature_dim, self.output_feature_dim)

        elif self.fusion_strategy == 'gated_fusion':
            # 策略3：门控融合
            self.gate_network = nn.Sequential(
                nn.Linear(self.gaf_feature_dim +
                          self.stat_feature_dim, self.output_feature_dim),
                nn.Sigmoid()
            )
            self.transform_network = nn.Sequential(
                nn.Linear(self.gaf_feature_dim +
                          self.stat_feature_dim, self.output_feature_dim),
                nn.Tanh()
            )

        elif self.fusion_strategy == 'residual_fusion':
            # 策略4：残差融合
            self.stat_transform = nn.Sequential(
                nn.Linear(self.stat_feature_dim, self.gaf_feature_dim),
                nn.ReLU(),
                nn.Linear(self.gaf_feature_dim, self.gaf_feature_dim)
            )
            self.output_projection = nn.Linear(
                self.gaf_feature_dim, self.output_feature_dim) if self.output_feature_dim != self.gaf_feature_dim else nn.Identity()

        elif self.fusion_strategy == 'cross_attention':
            # 策略5：交叉注意力融合
            self.gaf_to_stat_attention = nn.MultiheadAttention(
                self.gaf_feature_dim, num_heads=4, batch_first=True
            )
            self.stat_to_gaf_attention = nn.MultiheadAttention(
                self.stat_feature_dim, num_heads=4, batch_first=True
            )
            # 添加 LayerNorm（注意：需要与对应特征维度一致）
            self.gaf_norm = nn.LayerNorm(self.gaf_feature_dim)
            self.stat_norm = nn.LayerNorm(self.stat_feature_dim)

            self.output_projection = nn.Sequential(
                nn.Linear(self.gaf_feature_dim +
                          self.stat_feature_dim, self.output_feature_dim),
                nn.LayerNorm(self.output_feature_dim)
            )
            # 新增：FFN 模块
            self.ffn = nn.Sequential(
                nn.Linear(self.output_feature_dim,
                          self.output_feature_dim * 2),
                nn.GELU(),
                nn.Linear(self.output_feature_dim * 2, self.output_feature_dim)
            )
            self.ffn_norm = nn.LayerNorm(self.output_feature_dim)

        elif self.fusion_strategy == 'adaptive_fusion':
            # 策略6：自适应融合
            self.adaptation_network = nn.Sequential(
                nn.Linear(self.gaf_feature_dim +
                          self.stat_feature_dim, self.output_feature_dim),
                nn.ReLU(),
                nn.Linear(self.output_feature_dim, 2),  # 输出两个权重
                nn.Softmax(dim=-1)
            )
            self.gaf_projection = nn.Linear(
                self.gaf_feature_dim, self.output_feature_dim)
            self.stat_projection = nn.Linear(
                self.stat_feature_dim, self.output_feature_dim)

        else:
            raise ValueError(
                f"Unknown fusion strategy: {self.fusion_strategy}")

    def forward(self, gaf_features, stat_features):
        """前向传播

        Args:
            gaf_features: [B, C, gaf_feature_dim] GAF特征
            stat_features: [B, C, stat_feature_dim] 信号级统计特征

        Returns:
            fused_features: [B, C, output_feature_dim] 融合后的特征
        """
        B, C, _ = gaf_features.shape

        if self.fusion_strategy == 'concat_project':
            # 直接拼接后投影
            # [B, C, gaf_dim + stat_dim]
            concatenated = torch.cat([gaf_features, stat_features], dim=-1)
            fused_features = self.fusion_layer(
                concatenated)  # [B, C, output_dim]

        elif self.fusion_strategy == 'attention_fusion':
            # 注意力融合
            # [B, C, gaf_dim + stat_dim]
            concatenated = torch.cat([gaf_features, stat_features], dim=-1)

            # 计算注意力权重
            gaf_attention = self.gaf_attention(concatenated)  # [B, C, gaf_dim]
            stat_attention = self.stat_attention(
                concatenated)  # [B, C, stat_dim]

            # 应用注意力权重
            attended_gaf = gaf_features * gaf_attention  # [B, C, gaf_dim]
            attended_stat = stat_features * stat_attention  # [B, C, stat_dim]

            # 拼接并投影
            # [B, C, gaf_dim + stat_dim]
            attended_concat = torch.cat([attended_gaf, attended_stat], dim=-1)
            fused_features = self.output_projection(
                attended_concat)  # [B, C, output_dim]

        elif self.fusion_strategy == 'gated_fusion':
            # 门控融合
            # [B, C, gaf_dim + stat_dim]
            concatenated = torch.cat([gaf_features, stat_features], dim=-1)

            gate = self.gate_network(concatenated)  # [B, C, output_dim]
            transform = self.transform_network(
                concatenated)  # [B, C, output_dim]

            fused_features = gate * transform  # [B, C, output_dim]

        elif self.fusion_strategy == 'residual_fusion':
            # 残差融合
            transformed_stat = self.stat_transform(
                stat_features)  # [B, C, gaf_dim]
            residual_features = gaf_features + \
                transformed_stat  # [B, C, gaf_dim]
            fused_features = self.output_projection(
                residual_features)  # [B, C, output_dim]

        elif self.fusion_strategy == 'cross_attention':
            # 交叉注意力融合
            # GAF特征关注统计特征
            gaf_attended, _ = self.gaf_to_stat_attention(
                gaf_features, stat_features, stat_features
            )  # [B, C, gaf_dim]
            # 残差连接 + LayerNorm
            gaf_attended = self.gaf_norm(
                gaf_attended + gaf_features)  # [B, C, gaf_dim]

            # 统计特征关注GAF特征
            stat_attended, _ = self.stat_to_gaf_attention(
                stat_features, gaf_features, gaf_features
            )  # [B, C, stat_dim]
            # 残差连接 + LayerNorm
            stat_attended = self.stat_norm(
                stat_attended + stat_features)  # [B, C, stat_dim]

            # 拼接并投影
            # [B, C, gaf_dim + stat_dim]
            cross_attended = torch.cat([gaf_attended, stat_attended], dim=-1)
            fused_features = self.output_projection(
                cross_attended)  # [B, C, output_dim]

            # 新增：FFN 处理
            ffn_output = self.ffn(fused_features)
            fused_features = self.ffn_norm(
                ffn_output + fused_features)  # 残差 + 归一化

        elif self.fusion_strategy == 'adaptive_fusion':
            # 自适应融合
            # [B, C, gaf_dim + stat_dim]
            concatenated = torch.cat([gaf_features, stat_features], dim=-1)

            # 计算自适应权重
            fusion_weights = self.adaptation_network(concatenated)  # [B, C, 2]

            # 投影到相同维度
            projected_gaf = self.gaf_projection(
                gaf_features)  # [B, C, output_dim]
            projected_stat = self.stat_projection(
                stat_features)  # [B, C, output_dim]

            # 自适应加权
            fused_features = (fusion_weights[:, :, 0:1] * projected_gaf +
                              fusion_weights[:, :, 1:2] * projected_stat)  # [B, C, output_dim]

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
        self.conv1 = nn.Conv1d(
            2, 1, kernel_size, padding=kernel_size//2, bias=False)
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
            self.self_attention = nn.MultiheadAttention(
                feature_dim, num_heads=8, batch_first=True)
            self.norm1 = nn.LayerNorm(feature_dim)  # 自注意力后的归一化
            self.ffn = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.GELU(),
                nn.Linear(feature_dim * 2, feature_dim)
            )
            self.norm2 = nn.LayerNorm(feature_dim)

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
            attn_out, _ = self.self_attention(
                x_reshaped, x_reshaped, x_reshaped)
            # 残差连接 + LayerNorm
            out = self.norm1(attn_out + x)  # [B, C, feature_dim]
            out = self.ffn(out)
            out = self.norm2(out + out)
            return out


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
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms is not None:
                # 【修复】移除危险的动态模式切换，使用安全的批归一化策略
                if x.size(0) == 1:
                    # 对于batch size为1的情况，使用实例归一化或禁用BatchNorm
                    # 这样可以避免BatchNorm在训练模式下的数值不稳定问题
                    if self.training:
                        # 训练模式下：跳过BatchNorm以避免内存错误
                        pass  # 跳过BatchNorm
                    else:
                        # 评估模式下：正常使用BatchNorm
                        x = self.batch_norms[i](x)
                else:
                    # 正常batch size：直接使用BatchNorm
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
        # 'large_kernel', 'inception', 'dilated', 'multiscale'
        extractor_type='large_kernel',
        # 融合模块配置
        fusion_type='adaptive',  # 'adaptive', 'concat', 'add', 'mul', 'weighted_add'
        # 注意力模块配置
        attention_type='channel',  # 'channel', 'spatial', 'cbam', 'self', 'none'
        # 分类器配置
        classifier_type='mlp',  # 'mlp', 'simple'
        # 统计特征配置
        use_statistical_features=True,  # 是否使用统计特征
        stat_type='comprehensive',  # 'basic', 'comprehensive', 'correlation_focused'
        # 'concat', 'attention', 'gated', 'adaptive'
        multimodal_fusion_strategy='concat',

        # 信号级统计特征配置（新增）
        use_signal_level_stats=False,  # 是否使用信号级统计特征（对比实验）
        signal_stat_type='comprehensive',  # 'basic', 'comprehensive', 'extended'
        # 'concat_project', 'attention_fusion', 'gated_fusion', 'residual_fusion', 'cross_attention', 'adaptive_fusion'
        signal_stat_fusion_strategy='concat_project',
        signal_stat_feature_dim=32,  # 信号级统计特征的维度

        # 消融实验开关
        use_diff_branch=True,  # 是否使用diff分支进行融合（消融实验）
        
        # 通道压缩配置（新增）
        use_channel_compression=False,  # 是否使用通道压缩
        compression_strategy='conv1d',  # 'conv1d', 'grouped', 'separable', 'multiscale', 'attention_guided', 'adaptive', 'hvac_grouped'
        compression_ratio=0.7,  # 压缩比例
        adaptive_compression_ratios=[0.5, 0.7, 0.8],  # 自适应压缩可选比例
        hvac_group_compression_ratios=None,  # HVAC分组压缩比例
        compression_channels=None,  # 压缩通道数
        
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
        # 信号级统计特征配置
        self.use_signal_level_stats = use_signal_level_stats
        self.signal_stat_type = signal_stat_type
        self.signal_stat_fusion_strategy = signal_stat_fusion_strategy
        self.signal_stat_feature_dim = signal_stat_feature_dim

        # 消融实验开关
        self.use_diff_branch = use_diff_branch

        # 通道压缩配置
        self.use_channel_compression = use_channel_compression
        self.compression_strategy = compression_strategy
        self.compression_ratio = compression_ratio
        self.compression_channels = compression_channels
        self.adaptive_compression_ratios = adaptive_compression_ratios
        self.hvac_group_compression_ratios = hvac_group_compression_ratios

        # HVAC分组配置
        self.hvac_groups = hvac_groups
        self.feature_columns = feature_columns if feature_columns else []
        self.use_grouping = hvac_groups is not None

        # 智能推荐特征提取器（如果用户没有指定）
        if extractor_type == 'auto':
            recommended_extractor = self._get_recommended_extractor_for_signal_count(
                num_images)
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

        # 创建信号级统计特征提取器（如果启用）
        if self.use_signal_level_stats:
            self._build_signal_level_statistical_extractor()
            self._build_signal_level_statistical_fusion()
            
        # 创建通道压缩模块（如果启用）- 必须在注意力模块之前
        if self.use_channel_compression:
            self._build_channel_compression_module()
            
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
        print(f"  - 使用信号级统计特征: {use_signal_level_stats}")
        if use_signal_level_stats:
            print(f"  - 信号级统计特征类型: {signal_stat_type}")
            print(f"  - 信号级统计特征维度: {signal_stat_feature_dim}")
            print(f"  - 信号级统计融合策略: {signal_stat_fusion_strategy}")
        print(f"  - 使用分组: {self.use_grouping}")

        # 消融实验状态提示
        ablation_info = []
        if not self.use_diff_branch:
            ablation_info.append("GAF差分分支消融")
        if not self.use_statistical_features:
            ablation_info.append("统计特征消融")
        if self.use_signal_level_stats:
            ablation_info.append("信号级统计特征启用")
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
                self.sum_extractors[f'group_{group_idx}'] = self._create_extractor(
                )
                if self.use_diff_branch:
                    self.diff_extractors[f'group_{group_idx}'] = self._create_extractor(
                    )

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
            # if self.use_diff_branch:
            #     self.diff_extractor = self._create_extractor()
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

    def _build_signal_level_statistical_extractor(self):
        """构建信号级统计特征提取器"""
        self.signal_level_statistical_extractor = SignalLevelStatisticalExtractor(
            num_signals=self.num_images,
            time_length=self.time_length,
            stat_feature_dim=self.signal_stat_feature_dim,
            stat_type=self.signal_stat_type
        )

    def _build_signal_level_statistical_fusion(self):
        """构建信号级统计特征融合模块"""
        self.signal_level_statistical_fusion = SignalLevelStatisticalFusion(
            gaf_feature_dim=self.feature_dim,
            stat_feature_dim=self.signal_stat_feature_dim,
            output_feature_dim=self.feature_dim,
            fusion_strategy=self.signal_stat_fusion_strategy
        )

    def _build_attention_module(self):
        """构建注意力模块"""
        if self.attention_type == 'none':
            self.attention = nn.Identity()
        else:
            # 确定注意力模块应该使用的信号数量
            if self.use_channel_compression:
                if hasattr(self, 'compressed_num_images'):
                    # 使用动态计算的压缩后通道数
                    num_signals = self.compressed_num_images
                elif self.compression_channels is not None:
                    # 使用手动指定的压缩通道数
                    num_signals = self.compression_channels
                else:
                    # 使用压缩比例计算
                    num_signals = max(1, int(self.num_images * self.compression_ratio))
            else:
                # 没有压缩，使用原始通道数
                num_signals = self.num_images
                
            self.attention = SignalAttention(
                self.feature_dim,
                num_signals,
                self.attention_type
            )

    def _build_channel_compression_module(self):
        """构建通道压缩模块"""
        if self.compression_strategy == 'adaptive':
            # 自适应压缩
            min_output_channels = max(1, int(self.num_images * min(self.adaptive_compression_ratios)))
            self.channel_compressor = AdaptiveChannelCompressionModule(
                input_channels=self.num_images,
                min_output_channels=min_output_channels,
                feature_dim=self.feature_dim,
                compression_ratios=self.adaptive_compression_ratios
            )
            # 更新压缩后的通道数（取最小值）
            self.compressed_num_images = min_output_channels
            
        elif self.compression_strategy == 'hvac_grouped':
            # HVAC分组压缩
            self.channel_compressor = HVACSignalGroupCompressionModule(
                input_channels=self.num_images,
                feature_dim=self.feature_dim,
                hvac_groups=self.hvac_groups,
                feature_columns=self.feature_columns,
                group_compression_ratios=self.hvac_group_compression_ratios
            )
            # 更新压缩后的通道数
            self.compressed_num_images = self.channel_compressor.total_output_channels
        elif self.compression_strategy == 'signal_compression':
            output_channels = max(1, int(self.num_images * self.compression_ratio))
            self.channel_compressor = SignalCompressionModule(
                input_channels=self.num_images,
                output_channels=output_channels,
                feature_dim=self.feature_dim
            )
            self.compressed_num_images = output_channels
        else:
            # 标准压缩策略
            output_channels = max(1, int(self.num_images * self.compression_ratio))
            self.channel_compressor = ChannelCompressionModule(
                input_channels=self.num_images,
                output_channels=output_channels,
                feature_dim=self.feature_dim,
                compression_strategy=self.compression_strategy
            )
            # 更新压缩后的通道数
            self.compressed_num_images = output_channels
            
        print(f"通道压缩配置:")
        print(f"  - 压缩前通道数: {self.num_images}")
        print(f"  - 压缩后通道数: {self.compressed_num_images}")
        print(f"  - 实际压缩比例: {self.compressed_num_images / self.num_images:.2f}")
        print(f"  - 参数量减少估计: {1 - (self.compressed_num_images / self.num_images):.1%}")

    def _build_classifier(self, num_classes):
        """构建分类器"""
        # 根据是否使用通道压缩决定最终特征维度
        if self.use_channel_compression:
            final_feature_dim = self.feature_dim * self.compressed_num_images
        else:
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
                hidden_dims=None,
                block_type='basic',
                dropout=0.15,
                use_batch_norm=False
            )
            print(
                f"🏗️ 构建残差分类器 (基础残差块): {final_feature_dim} -> [1024, 512, 256] -> {num_classes}")
        elif self.classifier_type == 'residual_bottleneck':
            # 瓶颈残差分类器（适用于高维特征）
            self.classifier = ResidualClassifier(
                input_dim=final_feature_dim,
                num_classes=num_classes,
                hidden_dims=[2048, 1024, 512] if final_feature_dim > 1000 else [
                    1024, 512, 256],
                block_type='bottleneck',
                dropout=0.15,
                use_batch_norm=True
            )
            hidden_info = "[2048, 1024, 512]" if final_feature_dim > 1000 else "[1024, 512, 256]"
            print(
                f"🏗️ 构建瓶颈残差分类器 (带BatchNorm): {final_feature_dim} -> {hidden_info} -> {num_classes}")
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
            print(
                f"🏗️ 构建密集残差分类器 (最强表达能力): {final_feature_dim} -> [1024, 512, 256, 128] -> {num_classes}")
        else:
            raise ValueError(f"不支持的分类器类型: {self.classifier_type}")

        print(
            f"✅ 分类器构建完成: 类型={self.classifier_type}, 输入维度={final_feature_dim}, 输出类别={num_classes}")
    
    def print_model_structure(self, input_shape=None, detailed=True):
        """
        打印模型结构信息
        
        Args:
            input_shape: tuple, 输入数据形状 (B, C, H, W)，用于计算详细信息
            detailed: bool, 是否显示详细的参数信息
        """
        print("\n" + "="*80)
        print("🏗️ DualGAFNet 模型结构")
        print("="*80)
        
        # 1. 模型基本信息
        print(f"📊 模型配置:")
        print(f"   信号数量: {self.num_images}")
        print(f"   特征维度: {self.feature_dim}")
        print(f"   特征提取器: {self.extractor_type}")
        print(f"   融合类型: {self.fusion_type}")
        print(f"   注意力类型: {self.attention_type}")
        print(f"   分类器类型: {self.classifier_type}")
        print(f"   统计特征: {self.use_statistical_features}")
        print(f"   信号级统计: {self.use_signal_level_stats}")
        if hasattr(self, 'use_channel_compression'):
            print(f"   通道压缩: {self.use_channel_compression}")
            if self.use_channel_compression:
                print(f"     压缩策略: {self.compression_strategy}")
                print(f"     压缩比例: {self.compression_ratio}")
                if hasattr(self, 'compressed_num_images'):
                    print(f"     压缩后通道: {self.compressed_num_images}")
        
        # 2. 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📈 参数统计:")
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / (1024**2):.2f} MB")
        
        # 3. 各模块参数分布
        if detailed:
            print(f"\n🔍 各模块参数分布:")
            
            # 特征提取器
            if hasattr(self, 'sum_extractor'):
                sum_params = sum(p.numel() for p in self.sum_extractor.parameters())
                print(f"   Sum特征提取器: {sum_params:,} ({sum_params/total_params*100:.1f}%)")
            
            if hasattr(self, 'diff_extractor'):
                diff_params = sum(p.numel() for p in self.diff_extractor.parameters())
                print(f"   Diff特征提取器: {diff_params:,} ({diff_params/total_params*100:.1f}%)")
            
            # 融合模块
            if hasattr(self, 'fusion'):
                fusion_params = sum(p.numel() for p in self.fusion.parameters())
                print(f"   融合模块: {fusion_params:,} ({fusion_params/total_params*100:.1f}%)")
            
            # 统计特征
            if hasattr(self, 'statistical_extractor'):
                stat_params = sum(p.numel() for p in self.statistical_extractor.parameters())
                print(f"   统计特征提取器: {stat_params:,} ({stat_params/total_params*100:.1f}%)")
            
            if hasattr(self, 'multimodal_fusion'):
                mm_params = sum(p.numel() for p in self.multimodal_fusion.parameters())
                print(f"   多模态融合: {mm_params:,} ({mm_params/total_params*100:.1f}%)")
            
            # 通道压缩
            if hasattr(self, 'channel_compressor') and self.channel_compressor is not None:
                comp_params = sum(p.numel() for p in self.channel_compressor.parameters())
                print(f"   通道压缩: {comp_params:,} ({comp_params/total_params*100:.1f}%)")
            
            # 注意力模块
            if hasattr(self, 'attention'):
                att_params = sum(p.numel() for p in self.attention.parameters())
                print(f"   注意力模块: {att_params:,} ({att_params/total_params*100:.1f}%)")
            
            # 分类器
            if hasattr(self, 'classifier'):
                cls_params = sum(p.numel() for p in self.classifier.parameters())
                print(f"   主分类器: {cls_params:,} ({cls_params/total_params*100:.1f}%)")
        
        # 4. 模块层次结构
        print(f"\n🏛️ 模型层次结构:")
        for name, module in self.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            print(f"   {name}: {type(module).__name__} ({module_params:,} 参数)")
        
        # 5. 如果提供了输入形状，计算各层输出形状
        if input_shape is not None:
            self._print_layer_shapes(input_shape)
        
        print("="*80)
    
    def _print_layer_shapes(self, input_shape):
        """
        打印各层的输出形状（需要实际前向传播）
        
        Args:
            input_shape: tuple, (B, C, H, W)
        """
        print(f"\n📐 层输出形状分析 (输入: {input_shape}):")
        
        try:
            device = next(self.parameters()).device
            
            # 创建测试数据
            B, C, H, W = input_shape
            sum_x = torch.randn(B, C, H, W).to(device)
            diff_x = torch.randn(B, C, H, W).to(device)
            
            # 如果使用统计特征，创建时序数据
            if self.use_statistical_features:
                time_series_x = torch.randn(B, H, C).to(device)  # 假设时序长度等于H
            else:
                time_series_x = None
            
            self.eval()
            with torch.no_grad():
                # 这里可以添加hook来捕获中间层输出
                print(f"   输入 - Sum GAF: {sum_x.shape}")
                print(f"   输入 - Diff GAF: {diff_x.shape}")
                if time_series_x is not None:
                    print(f"   输入 - 时序数据: {time_series_x.shape}")
                
                # 前向传播并获取输出形状
                if time_series_x is not None:
                    output = self(sum_x, diff_x, time_series_x)
                else:
                    output = self(sum_x, diff_x)
                
                if isinstance(output, tuple):
                    print(f"   输出 - 主输出: {output[0].shape}")
                    print(f"   输出 - 辅助输出: {output[1].shape}")
                else:
                    print(f"   输出: {output.shape}")
                    
        except Exception as e:
            print(f"   ⚠️  无法计算层形状: {str(e)}")
    
    def get_model_summary(self):
        """
        获取模型摘要信息（字典格式）
        
        Returns:
            dict: 包含模型摘要信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = {
            'model_name': 'DualGAFNet',
            'signal_count': self.num_images,
            'feature_dim': self.feature_dim,
            'extractor_type': self.extractor_type,
            'fusion_type': self.fusion_type,
            'attention_type': self.attention_type,
            'classifier_type': self.classifier_type,
            'use_statistical_features': self.use_statistical_features,
            'use_signal_level_stats': self.use_signal_level_stats,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2),
            'compression_enabled': getattr(self, 'use_channel_compression', False)
        }
        
        if hasattr(self, 'use_channel_compression') and self.use_channel_compression:
            summary.update({
                'compression_strategy': self.compression_strategy,
                'compression_ratio': self.compression_ratio,
                'compressed_channels': getattr(self, 'compressed_num_images', 'unknown')
            })
        
        return summary

    def _extract_features(self, x, extractors_dict, default_extractor):
        """提取特征的通用方法"""
        B, C, H, W = x.shape

        if self.use_grouping:
            # 使用分组特征提取
            feats_list = []

            # 按组批量处理
            for group_idx in range(self.num_groups):
                group_channels = [
                    ch for ch, g in self.channel_to_group.items() if g == group_idx]

                if group_channels:
                    group_x = x[:, group_channels, :, :]
                    group_x = group_x.view(B * len(group_channels), 1, H, W)

                    extractor = extractors_dict[f'group_{group_idx}']
                    group_feats = extractor(group_x)
                    group_feats = group_feats.view(B, len(group_channels), -1)

                    for i, ch in enumerate(group_channels):
                        feats_list.append((ch, group_feats[:, i, :]))

            # 处理未分组的通道
            ungrouped_channels = [
                ch for ch, g in self.channel_to_group.items() if g == -1]
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
            sum_feats = self._extract_features(
                sum_x, self.sum_extractors, None)
        else:
            sum_feats = self._extract_features(sum_x, None, self.sum_extractor)

        # 标准双路GAF：提取diff分支特征并融合
        if self.use_grouping:
            diff_feats = self._extract_features(
                diff_x, self.diff_extractors, None)
        else:
            diff_feats = self._extract_features(
                diff_x, None, self.diff_extractor)
        # 根据是否使用diff分支决定特征融合策略
        if self.use_diff_branch:

            # 特征融合
            fused_feats = []
            for i in range(C):
                fused_feat = self.fusion(
                    sum_feats[:, i, :], diff_feats[:, i, :])
                fused_feats.append(fused_feat)

            # 所有融合类型现在都输出标准的feature_dim维度
            fused_feats = torch.stack(
                fused_feats, dim=1)  # [B, C, feature_dim]
        else:
            # 消融实验：仅使用sum分支，不进行融合
            fused_feats = sum_feats  # [B, C, feature_dim]

        # 通道压缩（如果启用）
        if self.use_channel_compression:
            if self.compression_strategy == 'signal_compression':
                # SignalCompressionModule 返回 (compressed, attention_weights)
                fused_feats, _ = self.channel_compressor(fused_feats)  # [B, compressed_C, feature_dim]
            else:
                # 其他压缩模块返回单个tensor
                fused_feats = self.channel_compressor(fused_feats)  # [B, compressed_C, feature_dim]

        # 如果使用统计特征，进行多模态融合
        if self.use_statistical_features:
            # 提取统计特征
            stat_features = self.statistical_extractor(
                time_series_x)  # [B, feature_dim]

            # 多模态融合
            fused_feats = self.multimodal_fusion(
                fused_feats, stat_features)  # [B, C, feature_dim]

        # 如果使用信号级统计特征，进行信号级融合
        if self.use_signal_level_stats:
            # 提取信号级统计特征
            signal_stat_features = self.signal_level_statistical_extractor(
                time_series_x)  # [B, C, signal_stat_feature_dim]

            # 信号级统计特征融合
            fused_feats = self.signal_level_statistical_fusion(
                fused_feats, signal_stat_features)  # [B, C, feature_dim]
                
        # 信号注意力（如果attention_type='none'，则相当于Identity）
        attended_feats = self.attention(fused_feats)  # [B, C, feature_dim]

        # 展平用于分类
        merged = attended_feats.reshape(B, -1)  # [B, (compressed_)C * feature_dim]

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
        use_statistical_features = getattr(
            configs, 'use_statistical_features', True)
        stat_type = getattr(configs, 'stat_type', 'comprehensive')
        multimodal_fusion_strategy = getattr(
            configs, 'multimodal_fusion_strategy', 'concat')

        # 获取信号级统计特征配置
        use_signal_level_stats = getattr(
            configs, 'use_signal_level_stats', False)
        signal_stat_type = getattr(
            configs, 'signal_stat_type', 'comprehensive')
        signal_stat_fusion_strategy = getattr(
            configs, 'signal_stat_fusion_strategy', 'concat_project')
        signal_stat_feature_dim = getattr(
            configs, 'signal_stat_feature_dim', 32)

        # 获取消融实验配置
        use_diff_branch = getattr(configs, 'use_diff_branch', True)

        # 获取通道压缩配置
        use_channel_compression = getattr(configs, 'use_channel_compression', False)
        compression_strategy = getattr(configs, 'compression_strategy', 'conv1d')
        compression_ratio = getattr(configs, 'compression_ratio', 0.7)
        adaptive_compression_ratios = getattr(configs, 'adaptive_compression_ratios', [0.5, 0.7, 0.8])
        hvac_group_compression_ratios = getattr(configs, 'hvac_group_compression_ratios', None)
        compression_channels = getattr(configs, 'compression_channels', None)
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
            use_signal_level_stats=use_signal_level_stats,
            signal_stat_type=signal_stat_type,
            signal_stat_fusion_strategy=signal_stat_fusion_strategy,
            signal_stat_feature_dim=signal_stat_feature_dim,
            use_diff_branch=use_diff_branch,
            use_channel_compression=use_channel_compression,
            compression_strategy=compression_strategy,
            compression_ratio=compression_ratio,
            compression_channels=compression_channels,
            adaptive_compression_ratios=adaptive_compression_ratios,
            hvac_group_compression_ratios=hvac_group_compression_ratios,
            hvac_groups=hvac_groups,
            feature_columns=feature_columns
        )

    def forward(self, sum_x, diff_x, time_series_x=None):
        return self.model(sum_x, diff_x, time_series_x)
    
    def print_model_structure(self, input_shape=None, detailed=True):
        """
        打印模型结构的便捷方法
        
        Args:
            input_shape: tuple, 输入数据形状 (B, C, H, W)
            detailed: bool, 是否显示详细参数信息
        """
        return self.model.print_model_structure(input_shape, detailed)
    
    def get_model_summary(self):
        """
        获取模型摘要信息的便捷方法
        
        Returns:
            dict: 包含模型摘要信息的字典
        """
        return self.model.get_model_summary()


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
            print(
                f"  - 输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

            # 计算参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel()
                                   for p in model.parameters() if p.requires_grad)
            print(f"\n模型复杂度:")
            print(f"  - 总参数数量: {total_params:,}")
            print(f"  - 可训练参数: {trainable_params:,}")
            print(f"  - 模型大小估计: {total_params * 4 / (1024**2):.2f} MB")

            # 计算内存估计
            input_memory = (sum_x.numel() + diff_x.numel() +
                            time_series_x.numel()) * 4 / (1024**2)
            print(f"  - 单batch输入内存: {input_memory:.2f} MB")

            print(f"✅ 测试通过")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            print(f"错误类型: {type(e).__name__}")

    print(f"\n{'='*100}")
    print("测试完成")
    print("="*100)
