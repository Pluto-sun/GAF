import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DualGAFNet import BasicBlock
from models.ClusteredInception import InceptionNet
from models.inception import BasicConv2d


class SimpleGAFNet(nn.Module):
    """简单的GAF网络 - 用于消融实验
    
    直接使用传统CV架构处理多通道GAF图像，不进行特征融合
    适配DualGAF_DDAHU数据加载器，但只使用一种GAF类型
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        
        # 获取配置
        self.num_channels = configs.enc_in
        self.num_classes = configs.num_class
        self.backbone_type = getattr(configs, 'backbone_type', 'resnet18')
        self.use_sum_branch = getattr(configs, 'use_sum_branch', True)  # 选择使用sum还是diff分支
        self.input_size = getattr(configs, 'seq_len', 96)  # GAF图像尺寸
        
        # 创建backbone
        self.backbone = self._create_backbone()
        
        print(f"SimpleGAFNet构建完成:")
        print(f"  - 输入通道数: {self.num_channels}")
        print(f"  - 输出类别数: {self.num_classes}")
        print(f"  - 主干网络: {self.backbone_type}")
        print(f"  - 使用分支: {'sum' if self.use_sum_branch else 'diff'}")
        print(f"  - 输入尺寸: {self.input_size}x{self.input_size}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  - 总参数数量: {total_params:,}")
        
    def _create_backbone(self):
        """创建主干网络"""
        if self.backbone_type == 'resnet18':
            return self._create_resnet18()
        elif self.backbone_type == 'resnet34':
            return self._create_resnet34()
        elif self.backbone_type == 'resnet50':
            return self._create_resnet50()
        elif self.backbone_type == 'inception':
            return self._create_inception()
        elif self.backbone_type == 'simple_cnn':
            return self._create_simple_cnn()
        else:
            raise ValueError(f"不支持的主干网络类型: {self.backbone_type}")
    
    def _create_resnet18(self):
        """创建ResNet18主干 - 适配多通道GAF输入"""
        return nn.Sequential(
            # 第一层：适应多通道输入，减少下采样
            nn.Conv2d(self.num_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet层
            self._make_layer(64, 64, 2, stride=1),    # 不下采样
            self._make_layer(64, 128, 2, stride=2),   # 下采样
            self._make_layer(128, 256, 2, stride=2),  # 下采样
            self._make_layer(256, 512, 2, stride=2),  # 下采样
            
            # 分类头
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
    
    def _create_resnet34(self):
        """创建ResNet34主干 - 适配多通道GAF输入"""
        return nn.Sequential(
            # 第一层：适应多通道输入
            nn.Conv2d(self.num_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet层
            self._make_layer(64, 64, 3, stride=1),    # 不下采样
            self._make_layer(64, 128, 4, stride=2),   # 下采样
            self._make_layer(128, 256, 6, stride=2),  # 下采样
            self._make_layer(256, 512, 3, stride=2),  # 下采样
            
            # 分类头
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
    
    def _create_resnet50(self):
        """创建ResNet50主干 - 使用BottleneckBlock"""
        return nn.Sequential(
            # 第一层：适应多通道输入
            nn.Conv2d(self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet层 (使用BottleneckBlock会更复杂，这里简化为BasicBlock)
            self._make_layer(64, 64, 3, stride=1),
            self._make_layer(64, 128, 4, stride=2),
            self._make_layer(128, 256, 6, stride=2),
            self._make_layer(256, 512, 3, stride=2),
            
            # 分类头
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
    
    def _create_inception(self):
        """创建Inception主干 - 使用ClusteredInception中的InceptionNet"""
        return InceptionNet(
            num_classes=self.num_classes,
            in_channels=self.num_channels,
            aux_logits=False
        )
    
    def _create_simple_cnn(self):
        """创建简单的CNN主干 - 轻量级版本"""
        return nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(self.num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层卷积块
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积块
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第四层卷积块
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 分类头
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """创建ResNet层"""
        layers = []
        
        # 第一个块可能需要下采样
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # 后续块
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, sum_x, diff_x, time_series_x=None):
        """前向传播 - 适配DualGAF数据加载器
        
        Args:
            sum_x: [B, C, H, W] Summation GAF图像
            diff_x: [B, C, H, W] Difference GAF图像
            time_series_x: [B, C, T] 原始时序数据（忽略）
        
        Returns:
            logits: [B, num_classes] 分类logits
        """
        # 选择使用哪个分支的数据
        if self.use_sum_branch:
            x = sum_x
        else:
            x = diff_x
        
        # 通过主干网络
        out = self.backbone(x)
        
        return out


class Model(nn.Module):
    """SimpleGAFNet的包装类"""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.model = SimpleGAFNet(configs)
        
    def forward(self, sum_x, diff_x, time_series_x=None):
        return self.model(sum_x, diff_x, time_series_x)


if __name__ == "__main__":
    print("="*80)
    print("测试SimpleGAFNet - 纯CV架构处理多通道GAF图像")
    print("="*80)
    
    # 测试不同配置
    test_configs = [
        (120, 96, "resnet18", "ResNet18"),
        (120, 96, "resnet34", "ResNet34"),
        (120, 96, "inception", "Inception"),
        (120, 96, "simple_cnn", "Simple CNN"),
    ]
    
    for signal_count, gaf_size, backbone_type, backbone_name in test_configs:
        print(f"\n{'-'*50}")
        print(f"测试配置: {backbone_name}")
        print(f"信号数量: {signal_count}, GAF尺寸: {gaf_size}x{gaf_size}")
        print(f"{'-'*50}")
        
        # 创建测试数据
        B = 2  # 小批次测试
        C, H, W = signal_count, gaf_size, gaf_size
        sum_x = torch.randn(B, C, H, W)
        diff_x = torch.randn(B, C, H, W)
        time_series_x = torch.randn(B, gaf_size, C)
        
        # 创建配置
        configs = type("cfg", (), {
            "enc_in": C,
            "num_class": 6,
            "seq_len": gaf_size,
            "backbone_type": backbone_type,
            "use_sum_branch": True,
        })()
        
        try:
            model = Model(configs)
            
            print(f"\n输入数据:")
            print(f"  - Summation GAF: {sum_x.shape}")
            print(f"  - Difference GAF: {diff_x.shape}")
            print(f"  - Time Series: {time_series_x.shape}")
            
            # 测试forward过程
            with torch.no_grad():
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
            
            print(f"✅ 测试通过")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("测试完成")
    print("="*80) 