import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DNet(nn.Module):
    """
    1D卷积神经网络，专用于HVAC时间序列异常检测
    
    Args:
        in_channels (int): 输入通道数（特征数）
        seq_len (int): 输入序列长度
        num_classes (int): 分类类别数
        dropout_rate (float): Dropout比例，默认0.3
        use_batch_norm (bool): 是否使用BatchNorm，默认True
    """
    
    def __init__(self, in_channels, seq_len, num_classes, dropout_rate=0.3, use_batch_norm=True):
        super(CNN1DNet, self).__init__()
        
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # 第一层卷积 - 提取局部特征
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
        
        # 第二层卷积 - 进一步特征提取
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(128)
        
        # 第三层卷积 - 高级特征提取
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn3 = nn.BatchNorm1d(256)
            
        # 第四层卷积 - 深层特征提取
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn4 = nn.BatchNorm1d(512)
        
        # 池化层
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接层
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, features] 或 [batch_size, features, seq_len]
            
        Returns:
            输出张量，形状为 [batch_size, num_classes]
        """
        # 处理输入维度 - 确保为 [batch_size, features, seq_len]
        if x.dim() == 3:
            if self.num_classes!=x.shape[1]:  # [batch, seq_len, features]
                x = x.permute(0, 2, 1)   # -> [batch, features, seq_len]
        elif x.dim() == 2:
            # [batch, seq_len] -> [batch, 1, seq_len]
            # 检查是否与预期的输入通道数匹配
            if x.shape[1] == self.seq_len:
                # 假设是 [batch, seq_len] 格式，但只有一个特征
                x = x.unsqueeze(1)  # -> [batch, 1, seq_len]
            else:
                # 假设是 [batch, features] 格式，需要添加序列维度
                x = x.unsqueeze(-1)  # -> [batch, features, 1]
        
        # 第一层卷积块
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # 第二层卷积块
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # 第三层卷积块
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # 第四层卷积块
        x = self.conv4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = F.relu(x)
        
        # 全局平均池化
        x = self.pool(x)  # -> [batch, 512, 1]
        x = x.squeeze(-1)  # -> [batch, 512]
        
        # 全连接层
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class Model(nn.Module):
    """
    1D-CNN模型的包装类，适配训练框架
    
    该模型专门处理时间序列数据，当输入包含多个数据源时（sum_data, diff_data, time_series_data），
    只使用time_series_data进行训练，因为它包含了最原始和最完整的时间序列信息。
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # 从配置中获取参数
        self.in_channels = getattr(args, 'enc_in', 8)
        self.seq_len = getattr(args, 'seq_len', 96)
        self.num_classes = getattr(args, 'num_class', 5)
        
        # 1D-CNN特有参数
        self.dropout_rate = getattr(args, 'cnn_dropout_rate', 0.3)
        self.use_batch_norm = getattr(args, 'cnn_use_batch_norm', True)
        
        print(f"🔧 1D-CNN模型配置:")
        print(f"   输入通道数: {self.in_channels}")
        print(f"   序列长度: {self.seq_len}")
        print(f"   分类类别数: {self.num_classes}")
        print(f"   Dropout率: {self.dropout_rate}")
        print(f"   使用BatchNorm: {self.use_batch_norm}")
        
        # 创建1D-CNN网络
        self.cnn_net = CNN1DNet(
            in_channels=self.in_channels,
            seq_len=self.seq_len,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        )
    
    def forward(self, *args):
        """
        前向传播，自动适配不同的输入格式
        
        支持的输入格式：
        1. forward(time_series_data) - 单个时间序列数据
        2. forward(sum_data, diff_data, time_series_data) - 三个数据源，使用time_series_data
        """
        if len(args) == 1:
            # 单个输入：直接使用
            time_series_data = args[0]
        elif len(args) == 3:
            # 三个输入：使用第三个参数（time_series_data）
            sum_data, diff_data, time_series_data = args
        else:
            raise ValueError(f"1D-CNN模型不支持{len(args)}个输入参数，支持1个或3个输入")
        
        # 调用1D-CNN网络
        return self.cnn_net(time_series_data) 