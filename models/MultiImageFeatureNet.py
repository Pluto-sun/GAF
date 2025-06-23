import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
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


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.layer1 = BasicBlock(1, 32, stride=2)
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        # x: [N, 1, H, W]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ChannelEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/2 x W/2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(32, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


# 迁移InceptionNet
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, aux_logits: bool = False):
        super(InceptionNet, self).__init__()
        self.aux_logits = aux_logits
        self.Conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(192, num_classes)
    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.inception = InceptionNet(num_classes=feature_dim, in_channels=1, aux_logits=False)
    def forward(self, x):
        # x: [N, 1, H, W]
        return self.inception(x)

class LargeKernelDilatedFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        # x: [N, 1, H, W]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class NoPaddingLargeKernelFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=6, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=6, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        # x: [N, 1, 64, 64]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MultiScaleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=True):
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) if use_pool else nn.Identity()
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) if use_pool else nn.Identity()
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) if use_pool else nn.Identity()
        )
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
    def forward(self, x):
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        x_cat = torch.cat([x3, x5, x7], dim=1)
        x = self.fuse(x_cat)
        return x

class MultiScaleStackedFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=32, channels_list=[16, 16],):
        super().__init__()
        self.channels_list = channels_list
        self.feature_dim = feature_dim
        ms_layers = []
        in_c = 1
        for out_c in channels_list:
            ms_layers.append(MultiScaleConvLayer(in_c, out_c, use_pool=True))
            in_c = out_c
        self.ms_layers = nn.Sequential(*ms_layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels_list[-1], self.feature_dim)
    def forward(self, x):   
        x = self.ms_layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.channels_list[-1] != self.feature_dim:
            x = self.fc(x)
        return x

class MultiImageFeatureNet(nn.Module):
    def __init__(
        self, feature_dim=32, num_classes=4, num_images=30, nhead=4, num_layers=2, 
        channels_list=None, hvac_groups=None, feature_columns=None
    ):
        super().__init__()
        self.num_images = num_images
        self.feature_dim = feature_dim
        
        if channels_list is None:
            channels_list = [32, 64, 128]
            
        self.hvac_groups = hvac_groups
        self.feature_columns = feature_columns if feature_columns else []
        self.use_grouping = hvac_groups is not None
        
        if self.use_grouping:
            # 使用分组特征提取
            # 创建通道索引到组的映射
            self.channel_to_group = self._create_channel_mapping()
            
            # 为每个组创建特征提取器
            self.num_groups = len(self.hvac_groups)
            self.feature_extractors = nn.ModuleDict()
            
            for group_idx in range(self.num_groups):
                self.feature_extractors[f'group_{group_idx}'] = NoPaddingLargeKernelFeatureExtractor(feature_dim)
            
            # 如果有未分组的通道，创建一个默认特征提取器
            if -1 in self.channel_to_group.values():
                self.feature_extractors['default'] = NoPaddingLargeKernelFeatureExtractor(feature_dim)
            
            print(f"使用分组特征提取：创建了 {len(self.feature_extractors)} 个特征提取器")
            print(f"通道分组映射: {self.channel_to_group}")
        else:
            # 不使用分组，所有通道共用一个特征提取器
            self.feature_extractor = InceptionFeatureExtractor(feature_dim)
            print("使用默认特征提取：所有通道共用一个特征提取器")
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * self.num_images, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def _create_channel_mapping(self):
        """创建通道索引到组的映射"""
        channel_to_group = {}
        
        if not self.feature_columns:
            # 如果没有特征列信息，使用默认映射（所有通道使用第一个组）
            for i in range(self.num_images):
                channel_to_group[i] = 0
            return channel_to_group
        
        # 为每个通道找到对应的组
        for channel_idx, column_name in enumerate(self.feature_columns):
            group_found = False
            
            # 遍历所有组，查找匹配的列名
            for group_idx, group_signals in enumerate(self.hvac_groups):
                # 使用包含匹配而不是精确匹配，以处理列名变体
                if any(signal in column_name.upper() for signal in [s.upper() for s in group_signals]):
                    channel_to_group[channel_idx] = group_idx
                    group_found = True
                    break
            
            # 如果没有找到匹配的组，使用默认组（-1表示未分组）
            if not group_found:
                channel_to_group[channel_idx] = -1
                print(f"警告: 通道 '{column_name}' 未找到匹配的信号组，将使用默认特征提取器")
        
        return channel_to_group

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.num_images, f"输入通道数应为{self.num_images}，实际为{C}"
        
        # 将输入reshape为每个通道单独处理
        # x = x.view(B * C, 1, H, W)

        if self.use_grouping:
            # 使用分组特征提取（优化版本：按组批量处理）
            feats_list = []
            
            # 按组批量处理，提高效率
            for group_idx in range(self.num_groups):
                # 找到属于当前组的所有通道
                group_channels = [ch for ch, g in self.channel_to_group.items() if g == group_idx]
                
                if group_channels:
                    # 批量处理当前组的所有通道
                    group_x = x[:, group_channels, :, :]  # [B, group_size, H, W]
                    group_x = group_x.view(B * len(group_channels), 1, H, W)  # [B*group_size, 1, H, W]
                    
                    extractor = self.feature_extractors[f'group_{group_idx}']
                    group_feats = extractor(group_x)  # [B*group_size, feature_dim]
                    group_feats = group_feats.view(B, len(group_channels), -1)  # [B, group_size, feature_dim]
                    
                    # 将组内特征按原始通道顺序插入
                    for i, ch in enumerate(group_channels):
                        feats_list.append((ch, group_feats[:, i, :]))
            
            # 处理未分组的通道
            ungrouped_channels = [ch for ch, g in self.channel_to_group.items() if g == -1]
            if ungrouped_channels and 'default' in self.feature_extractors:
                for ch in ungrouped_channels:
                    channel_x = x[:, ch:ch+1, :, :]  # [B, 1, H, W]
                    channel_feat = self.feature_extractors['default'](channel_x)
                    feats_list.append((ch, channel_feat))
            
            # 按通道索引排序并堆叠
            feats_list.sort(key=lambda x: x[0])  # 按通道索引排序
            feats = torch.stack([feat for _, feat in feats_list], dim=1)  # [B, C, feature_dim]
        else:
            # 使用默认特征提取器处理所有通道
            x = x.view(B * C, 1, H, W)
            feats = self.feature_extractor(x)  # [B*C, feature_dim]
            feats = feats.view(B, C, -1)       # [B, C, feature_dim]
        
        # 使用Transformer编码序列特征
        feats_t = feats.permute(1, 0, 2)        # [C, B, feature_dim]
        feats_enc = self.transformer_encoder(feats_t)  # [C, B, feature_dim]
        feats_enc = feats_enc.permute(1, 0, 2)  # [B, C, feature_dim]
        
        # 合并所有特征进行最终分类
        merged = feats_enc.reshape(B, -1)       # [B, C * feature_dim]
        out = self.classifier(merged)
        return out


class Model(nn.Module):
    """
    多图像特征提取与分类网络
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        feature_dim = configs.feature_dim
        num_classes = configs.num_class
        num_images = configs.enc_in
        
        # 获取HVAC信号分组配置（如果存在）
        hvac_groups = getattr(configs, 'hvac_groups', None)
        feature_columns = getattr(configs, 'feature_columns', None)
        
        self.model = MultiImageFeatureNet(
            feature_dim=feature_dim, 
            num_classes=num_classes, 
            num_images=num_images,
            hvac_groups=hvac_groups,
            feature_columns=feature_columns
        )

    def forward(self, x_enc):
        return self.model(x_enc)


if __name__ == "__main__":
    # 测试分组特征提取
    B, C, H, W = 2, 26, 32, 32
    x = torch.randn(B, C, H, W)
    
    # 模拟HVAC特征列名
    feature_columns = [
        'SA_TEMP','OA_TEMP','MA_TEMP','RA_TEMP','ZONE_TEMP_1','ZONE_TEMP_2','ZONE_TEMP_3','ZONE_TEMP_4','ZONE_TEMP_5',     # 温度组
        'OA_CFM','RA_CFM','SA_CFM',                  # 流量组
        'SA_SP', 'SA_SPSPT',                 # 设定点组
        'SF_WAT', 'RF_WAT',        # 阀门组
        'SF_SPD','RF_SPD','SF_CS','RF_CS',        # 阀门组
        'CHWC_VLV_DM','CHWC_VLV',        # 阀门组
        'OA_DMPR_DM','RA_DMPR_DM','OA_DMPR','RA_DMPR'                     # 未知信号（测试默认组）
    ]
    
    # 创建配置
    configs = type("cfg", (), {
        "feature_dim": 32, 
        "num_class": 4, 
        "enc_in": C,
        "feature_columns": feature_columns,
        "hvac_groups": [
            ['SA_TEMP','OA_TEMP','MA_TEMP','RA_TEMP','ZONE_TEMP_1','ZONE_TEMP_2','ZONE_TEMP_3','ZONE_TEMP_4','ZONE_TEMP_5'],  # 温度组
            ['OA_CFM','RA_CFM','SA_CFM'],                 # 流量组
            ['SA_SP', 'SA_SPSPT'],                          # 设定点组
            ['SF_WAT', 'RF_WAT'],                  # 阀门组
            ['SF_SPD','RF_SPD','SF_CS','RF_CS'],                  # 阀门组
            ['CHWC_VLV_DM','CHWC_VLV'],                  # 阀门组
            ['OA_DMPR_DM','RA_DMPR_DM','OA_DMPR','RA_DMPR'],                  # 阀门组
        ]
    })()
    
    model = Model(configs)
    
    print(f"输入x shape: {x.shape}")
    print(f"特征列: {feature_columns}")
    print(f"通道分组映射: {model.model.channel_to_group}")
    
    # 测试forward过程
    out = model(x)
    print(f"最终输出 shape: {out.shape}")
    
    # 验证不同组的特征提取器确实是不同的
    print("\n验证特征提取器:")
    for name, extractor in model.model.feature_extractors.items():
        print(f"  {name}: {type(extractor).__name__}")
    
    print(f"\n模型总参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
