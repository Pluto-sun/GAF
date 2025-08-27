class EfficientClassifier(nn.Module):
    def __init__(self, num_sensors=110, sensor_feature_dim=128, num_classes=1000, 
                 compressed_feature_dim=32, compressed_sensors=64, 
                 intermediate_dim=256):
        """
        传感器信号分类器：特征优先压缩 + 传感器融合
        
        Args:
            num_sensors: 传感器数量 (110)
            sensor_feature_dim: 每个传感器特征维度 (128)
            num_classes: 分类类别数
            compressed_feature_dim: 压缩后的特征维度 (32)
            compressed_sensors: 压缩后的传感器数量 (64)
            intermediate_dim: 矩阵分解的中间维度
        """
        super(EfficientClassifier, self).__init__()
        
        self.num_sensors = num_sensors
        self.sensor_feature_dim = sensor_feature_dim
        self.compressed_feature_dim = compressed_feature_dim
        self.compressed_sensors = compressed_sensors
        
        # 阶段1: 单传感器特征压缩 (信号降维)
        # 每个传感器独立进行特征压缩: 128 → 32
        self.sensor_feature_compress = nn.Sequential(
            nn.Linear(sensor_feature_dim, compressed_feature_dim * 2),  # 128 → 64
            nn.BatchNorm1d(num_sensors),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(compressed_feature_dim * 2, compressed_feature_dim),  # 64 → 32
            nn.BatchNorm1d(num_sensors),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 阶段2: 传感器间注意力机制 (识别重要传感器)
        self.sensor_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化每个传感器的特征
            nn.Conv1d(num_sensors, num_sensors // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_sensors // 4, num_sensors, 1),
            nn.Sigmoid()
        )
        
        # 阶段3: 传感器融合 (跨传感器的信息融合)
        # 110个传感器 → 64个虚拟传感器
        self.sensor_fusion = nn.Sequential(
            nn.Linear(num_sensors, compressed_sensors * 2),  # 110 → 128
            nn.BatchNorm1d(compressed_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            
            nn.Linear(compressed_sensors * 2, compressed_sensors),  # 128 → 64
            nn.BatchNorm1d(compressed_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15)
        )
        
        # 阶段4: 矩阵分解的分类头
        final_dim = compressed_sensors * compressed_feature_dim  # 64 * 32 = 2048
        
        # 矩阵分解: 避免直接大矩阵乘法
        # 原始: 2048 → 1024 → num_classes 需要 2M+ 参数
        # 分解: 2048 → 256 → 1024 → num_classes 需要 0.52M + 0.26M 参数
        self.classifier_decomp = nn.Sequential(
            # 第一层分解 - 主要降维
            nn.Linear(final_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 第二层分解 - 特征重组
            nn.Linear(intermediate_dim, intermediate_dim * 2),
            nn.BatchNorm1d(intermediate_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 最终分类层
            nn.Linear(intermediate_dim * 2, num_classes)
        )
        
        # 权重初始化
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
        """
        传感器信号分类前向传播
        Args:
            x: [batch_size, num_sensors, sensor_feature_dim] = [B, 110, 128]
        Returns:
            output: [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # 阶段1: 单传感器特征压缩
        # 每个传感器独立压缩: [B, 110, 128] → [B, 110, 32]
        sensor_compressed = self.sensor_feature_compress(x)  # [B, 110, 32]
        
        # 阶段2: 传感器重要性注意力
        # 识别哪些传感器对当前样本更重要
        sensor_importance = self.sensor_attention(sensor_compressed.transpose(1, 2))  # [B, 110, 1]
        sensor_importance = sensor_importance.transpose(1, 2)  # [B, 1, 110]
        
        # 应用注意力权重
        weighted_sensors = sensor_compressed * sensor_importance  # [B, 110, 32]
        
        # 阶段3: 传感器融合
        # 跨传感器维度的信息融合: [B, 110, 32] → [B, 64, 32]
        # 转置后融合: [B, 32, 110] → [B, 32, 64] → [B, 64, 32]
        sensors_transposed = weighted_sensors.transpose(1, 2)  # [B, 32, 110]
        fused_sensors = self.sensor_fusion(sensors_transposed)  # [B, 32, 64]
        fused_sensors = fused_sensors.transpose(1, 2)  # [B, 64, 32]
        
        # 阶段4: 分类
        # 展平并通过分解的分类器
        flattened = fused_sensors.view(batch_size, -1)  # [B, 64*32=2048]
        output = self.classifier_decomp(flattened)  # [B, num_classes]
        
        return output
    
    def get_sensor_importance(self, x):
        """
        获取传感器重要性分数，用于分析哪些传感器更重要
        """
        with torch.no_grad():
            sensor_compressed = self.sensor_feature_compress(x)
            sensor_importance = self.sensor_attention(sensor_compressed.transpose(1, 2))
            return sensor_importance.squeeze(-1)  # [B, 110]
    
    def get_param_count(self):
        """计算参数量"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 分模块统计
        feature_compress_params = sum(p.numel() for p in self.sensor_feature_compress.parameters() if p.requires_grad)
        attention_params = sum(p.numel() for p in self.sensor_attention.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in self.sensor_fusion.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier_decomp.parameters() if p.requires_grad)
        
        print(f"传感器信号分类器参数统计:")
        print(f"  特征压缩模块: {feature_compress_params:,}")
        print(f"  传感器注意力: {attention_params:,}")
        print(f"  传感器融合模块: {fusion_params:,}")
        print(f"  分类器模块: {classifier_params:,}")
        print(f"  总计参数: {total_params:,}")
        
        return total_params