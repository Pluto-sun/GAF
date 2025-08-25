import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConvBlock(nn.Module):
    """
    多尺度1D卷积块：并行使用多个不同kernel_size的卷积提取局部时间特征
    输入: [batch_size, in_channels, feature_dim]
    输出: [batch_size, out_channels, feature_dim]
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7), dropout_rate=0.2):
        super(MultiScaleConvBlock, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels

        # 计算每个分支的输出通道（均分）
        per_branch_out = out_channels // len(kernel_sizes)
        # 修正：确保总通道数正确（避免整除导致少通道）
        remainder = out_channels - per_branch_out * len(kernel_sizes)

        self.branches = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # 分配通道：前面的分支多分一点余数
            branch_out = per_branch_out + (1 if i < remainder else 0)
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, branch_out, kernel_size=k,
                          padding=k//2, bias=False),
                nn.BatchNorm1d(branch_out),
                nn.GELU()
            ))

        # 融合卷积（可选，用于跨通道交互）
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        # Dropout
        self.dropout = nn.Dropout1d(dropout_rate)

    def forward(self, x):
        # 并行通过各分支
        residual = self.shortcut(x)
        branch_outputs = [branch(x) for branch in self.branches]
        # 拼接
        x = torch.cat(branch_outputs, dim=1)  # [B, total_out_ch, F]
        # 融合
        x = self.fusion_conv(x)
        # Dropout
        x = self.dropout(x)
        return x + residual  # ✅ 块内残差


class SignalCompressionModule(nn.Module):
    """
    信号压缩模块：注意力机制 + 1D卷积
    输入: [batch_size, channel, feature_dim]
    输出: [batch_size, new_channel_num, feature_dim]
    """

    def __init__(self,
                 input_channels,
                 output_channels,
                 feature_dim=128,
                 attention_heads=8,
                 conv_layers=3):
        super(SignalCompressionModule, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.feature_dim = feature_dim

        # 1. 通道注意力机制 - 学习通道重要性
        # 确保中间通道数至少为1
        mid_channels = max(1, input_channels // 4)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, C, F] -> [B, C, 1]
            nn.Conv1d(input_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, input_channels, 1),
            nn.Sigmoid()
        )

        # 2. 自注意力机制 - 学习通道间关系
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=attention_heads,
            batch_first=True
        )

        # 3. 改进的1D卷积压缩层 - 添加激活函数和dropout
        self.compression_blocks = nn.ModuleList()

        # 计算各层的通道数
        if conv_layers == 1:
            # 单层直接压缩
            layer_channels = [input_channels, output_channels]
        else:
            # 多层逐步压缩 - 需要 conv_layers + 1 个元素
            step = (input_channels - output_channels) // conv_layers
            layer_channels = [input_channels]

            # 生成中间层的通道数
            for i in range(conv_layers - 1):
                next_channels = max(
                    output_channels, input_channels - (i + 1) * step)
                layer_channels.append(next_channels)

            # 添加最后一层的输出通道数
            layer_channels.append(output_channels)

        print(f"信号压缩层通道配置: {layer_channels}")

        # 构建压缩块
        for i in range(conv_layers):
            in_channels = layer_channels[i]
            out_channels = layer_channels[i + 1]
            is_last_layer = (i == conv_layers - 1)

            # 最后一层使用不同配置
            dropout_rate = 0.1 if is_last_layer else 0.2
            activation = nn.Tanh() if is_last_layer else nn.GELU()

            # 创建多尺度卷积块
            block = MultiScaleConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=(3, 5, 7),  # 使用 3, 5, 7 多尺度卷积
                dropout_rate=dropout_rate
            )

            self.compression_blocks.append(block)

        # 4. 改进的残差连接
        self.residual_proj = None
        if input_channels != output_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv1d(input_channels, output_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm1d(output_channels)
            )

        # 5. 改进的特征增强层
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(output_channels, output_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
            nn.Dropout1d(0.1),
            nn.Conv1d(output_channels, output_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels)
        )

        # 6. 层归一化（可选，用于稳定训练）
        self.layer_norm = nn.LayerNorm([output_channels, feature_dim])

        # 7. 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        改进的权重初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 使用Kaiming初始化
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        改进的前向传播
        Args:
            x: [batch_size, channel, feature_dim]
        Returns:
            compressed: [batch_size, new_channel_num, feature_dim]
            attention_weights: 注意力权重用于可视化
        """
        batch_size, channels, features = x.shape
        identity = x  # 保存原始输入用于深度残差连接

        # 1. 通道注意力
        channel_att = self.channel_attention(x)  # [B, C, 1]
        x_att = x * channel_att  # 加权原始特征

        # 2. 自注意力机制 - 学习通道间关系
        att_out, attention_weights = self.self_attention(
            x_att, x_att, x_att  # [B, C, F]
        )

        # 3. 第一个残差连接
        x_enhanced = x_att + att_out

        # 4. 改进的1D卷积压缩
        compressed = x_enhanced
        intermediate_features = []  # 保存中间特征用于深度监督

        for i, block in enumerate(self.compression_blocks):
            compressed = block(compressed)
            intermediate_features.append(compressed)


        # 5. 主残差连接
        residual = identity
        if self.residual_proj is not None:
            residual = self.residual_proj(identity)
        else:
            residual = identity

        # 维度适配（如果需要）
        # if residual.shape[1] != compressed.shape[1]:
        #     # 使用自适应池化调整通道数
        #     residual = F.adaptive_avg_pool1d(
        #         residual.view(batch_size, residual.shape[1], -1),
        #         compressed.shape[1]
        #     ).view(batch_size, compressed.shape[1], features)

        compressed = compressed + residual

        # 6. 特征增强
        enhanced = self.feature_enhance(compressed)
        compressed = compressed + enhanced  # 残差连接

        # 7. 层归一化
        compressed = self.layer_norm(compressed)

        return compressed, attention_weights


class ChannelCompressionModule(nn.Module):
    """信号通道压缩模块

    通过一维卷积学习信号间的相关性，压缩冗余的信号特征

    支持多种压缩策略：
    - 'conv1d': 简单一维卷积
    - 'grouped': 分组卷积
    - 'separable': 深度可分离卷积
    - 'multiscale': 多尺度卷积融合
    - 'attention_guided': 注意力引导的压缩
    """

    def __init__(self,
                 input_channels,
                 output_channels,
                 feature_dim,
                 compression_strategy='conv1d',
                 kernel_size=3,
                 num_groups=4,
                 use_attention=True,
                 use_residual=True,
                 dropout=0.1):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.feature_dim = feature_dim
        self.compression_strategy = compression_strategy
        self.use_residual = use_residual

        # 压缩比例
        self.compression_ratio = output_channels / input_channels

        print(f"信号通道压缩模块初始化:")
        print(f"  - 输入通道数: {input_channels}")
        print(f"  - 输出通道数: {output_channels}")
        print(f"  - 特征维度: {feature_dim}")
        print(f"  - 压缩比例: {self.compression_ratio:.2f}")
        print(f"  - 压缩策略: {compression_strategy}")

        self._build_compression_module(
            kernel_size, num_groups, use_attention, dropout)

        # 如果使用残差连接，需要维度匹配
        if self.use_residual and input_channels != output_channels:
            self.residual_projection = nn.Conv1d(
                input_channels, output_channels, 1)
        else:
            self.residual_projection = nn.Identity()

    def _build_compression_module(self, kernel_size, num_groups, use_attention, dropout):
        """构建压缩模块"""

        if self.compression_strategy == 'conv1d':
            # 策略1：简单一维卷积
            padding = kernel_size // 2
            self.compression_layer = nn.Sequential(
                nn.Conv1d(self.input_channels, self.output_channels,
                          kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(self.output_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        elif self.compression_strategy == 'grouped':
            # 策略2：分组卷积
            padding = kernel_size // 2
            # 确保分组数能被输入和输出通道数整除
            max_groups = min(num_groups, self.input_channels,
                             self.output_channels)
            groups = 1
            for g in range(max_groups, 0, -1):
                if self.input_channels % g == 0 and self.output_channels % g == 0:
                    groups = g
                    break

            self.compression_layer = nn.Sequential(
                nn.Conv1d(self.input_channels, self.output_channels,
                          kernel_size, padding=padding, groups=groups, bias=False),
                nn.BatchNorm1d(self.output_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        elif self.compression_strategy == 'separable':
            # 策略3：深度可分离卷积
            padding = kernel_size // 2
            self.compression_layer = nn.Sequential(
                # 深度卷积
                nn.Conv1d(self.input_channels, self.input_channels,
                          kernel_size, padding=padding, groups=self.input_channels, bias=False),
                nn.BatchNorm1d(self.input_channels),
                nn.ReLU(inplace=True),
                # 点卷积
                nn.Conv1d(self.input_channels,
                          self.output_channels, 1, bias=False),
                nn.BatchNorm1d(self.output_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        elif self.compression_strategy == 'multiscale':
            # 策略4：多尺度卷积融合
            self.conv_branches = nn.ModuleList()
            scales = [1, 3, 5]  # 不同的kernel size
            branch_channels = self.output_channels // len(scales)

            for i, scale in enumerate(scales):
                padding = scale // 2
                if i == len(scales) - 1:  # 最后一个分支处理剩余通道
                    out_ch = self.output_channels - branch_channels * i
                else:
                    out_ch = branch_channels

                branch = nn.Sequential(
                    nn.Conv1d(self.input_channels, out_ch,
                              scale, padding=padding, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True)
                )
                self.conv_branches.append(branch)

            self.dropout = nn.Dropout(dropout)

        elif self.compression_strategy == 'attention_guided':
            # 策略5：注意力引导的压缩
            padding = kernel_size // 2

            # 通道注意力模块
            # 确保中间层至少有1个通道
            mid_channels = max(1, self.input_channels // 4)
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(self.input_channels, mid_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, self.input_channels, 1),
                nn.Sigmoid()
            )

            # 主压缩层
            self.compression_layer = nn.Sequential(
                nn.Conv1d(self.input_channels, self.output_channels,
                          kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(self.output_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        else:
            raise ValueError(
                f"Unknown compression strategy: {self.compression_strategy}")

        # 可选的后处理注意力
        if use_attention and self.compression_strategy != 'attention_guided':
            # 确保中间层至少有1个通道
            mid_channels = max(1, self.output_channels // 4)
            self.post_attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(self.output_channels, mid_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, self.output_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.post_attention = None

    def forward(self, x):
        """前向传播

        Args:
            x: [B, C, feature_dim] 输入特征

        Returns:
            compressed: [B, new_C, feature_dim] 压缩后的特征
        """
        B, C, D = x.shape

        # 转置以适应Conv1d: [B, C, D] -> [B, C, D] (C作为通道维度)

        if self.compression_strategy == 'multiscale':
            # 多尺度分支
            branch_outputs = []
            for branch in self.conv_branches:
                branch_out = branch(x)  # [B, branch_channels, D]
                branch_outputs.append(branch_out)

            # [B, output_channels, D]
            compressed = torch.cat(branch_outputs, dim=1)
            compressed = self.dropout(compressed)

        elif self.compression_strategy == 'attention_guided':
            # 注意力引导压缩
            # 计算通道注意力权重
            attention_weights = self.channel_attention(x)  # [B, C, 1]

            # 应用注意力权重
            attended_x = x * attention_weights  # [B, C, D]

            # 压缩
            compressed = self.compression_layer(
                attended_x)  # [B, output_channels, D]

        else:
            # 其他策略
            compressed = self.compression_layer(x)  # [B, output_channels, D]

        # 后处理注意力
        if self.post_attention is not None:
            post_attention_weights = self.post_attention(
                compressed)  # [B, output_channels, 1]
            compressed = compressed * post_attention_weights

        # 残差连接（如果维度匹配）
        if self.use_residual:
            residual = self.residual_projection(x)  # [B, output_channels, D]

            # 如果输出通道数小于输入通道数，对残差进行适应性处理
            if residual.size(1) > compressed.size(1):
                # 通过平均池化减少残差的通道数
                residual = F.adaptive_avg_pool1d(residual.transpose(
                    1, 2), compressed.size(1)).transpose(1, 2)
            elif residual.size(1) < compressed.size(1):
                # 通过重复扩展残差的通道数
                repeat_times = compressed.size(1) // residual.size(1)
                remainder = compressed.size(1) % residual.size(1)
                residual_expanded = residual.repeat(1, repeat_times, 1)
                if remainder > 0:
                    residual_expanded = torch.cat(
                        [residual_expanded, residual[:, :remainder, :]], dim=1)
                residual = residual_expanded

            compressed = compressed + residual

        return compressed


class AdaptiveChannelCompressionModule(nn.Module):
    """自适应信号通道压缩模块

    根据信号的重要性和相关性动态确定压缩策略
    """

    def __init__(self,
                 input_channels,
                 min_output_channels,
                 feature_dim,
                 compression_ratios=[0.5, 0.7, 0.8],
                 selection_strategy='importance_based'):
        super().__init__()

        self.input_channels = input_channels
        self.min_output_channels = min_output_channels
        self.feature_dim = feature_dim
        self.compression_ratios = compression_ratios
        self.selection_strategy = selection_strategy

        # 为每个压缩比例创建压缩模块
        self.compression_modules = nn.ModuleDict()
        for i, ratio in enumerate(compression_ratios):
            output_channels = max(min_output_channels,
                                  int(input_channels * ratio))
            # 使用索引而不是小数作为键名
            self.compression_modules[f'ratio_{i}'] = ChannelCompressionModule(
                input_channels, output_channels, feature_dim,
                compression_strategy='attention_guided'
            )

        # 选择网络：决定使用哪个压缩比例
        self.selection_network = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, len(compression_ratios)),
            nn.Softmax(dim=1)
        )

        print(f"自适应信号通道压缩模块初始化:")
        print(f"  - 输入通道数: {input_channels}")
        print(f"  - 最小输出通道数: {min_output_channels}")
        print(f"  - 可选压缩比例: {compression_ratios}")
        print(f"  - 选择策略: {selection_strategy}")

    def forward(self, x):
        """前向传播

        Args:
            x: [B, C, feature_dim] 输入特征

        Returns:
            compressed: [B, adaptive_C, feature_dim] 自适应压缩后的特征
        """
        B, C, D = x.shape

        # 计算选择权重
        selection_weights = self.selection_network(x)  # [B, num_ratios]

        # 获取所有压缩结果
        compressed_outputs = []
        for i, ratio in enumerate(self.compression_ratios):
            module_key = f'ratio_{i}'  # 使用索引作为键名
            compressed = self.compression_modules[module_key](
                x)  # [B, output_channels_i, D]
            compressed_outputs.append(compressed)

        # 根据权重进行加权融合
        # 注意：不同压缩比例的输出通道数可能不同，需要统一维度

        # 方法1：选择最高权重的压缩结果
        if self.selection_strategy == 'hard_selection':
            selected_idx = torch.argmax(selection_weights, dim=1)  # [B]
            batch_results = []
            for b in range(B):
                batch_results.append(
                    compressed_outputs[selected_idx[b]][b:b+1])
            compressed = torch.cat(batch_results, dim=0)

        # 方法2：基于重要性的软融合
        else:  # importance_based
            # 将所有压缩结果插值到相同的通道数（使用最小通道数）
            target_channels = min([out.size(1) for out in compressed_outputs])

            weighted_sum = None
            for i, compressed_out in enumerate(compressed_outputs):
                # 如果通道数不匹配，进行自适应平均池化
                if compressed_out.size(1) != target_channels:
                    compressed_out = F.adaptive_avg_pool1d(
                        compressed_out.transpose(1, 2), target_channels
                    ).transpose(1, 2)

                # 加权累加
                weight = selection_weights[:, i:i+1].unsqueeze(-1)  # [B, 1, 1]
                if weighted_sum is None:
                    weighted_sum = weight * compressed_out
                else:
                    weighted_sum += weight * compressed_out

            compressed = weighted_sum

        return compressed


class HVACSignalGroupCompressionModule(nn.Module):
    """HVAC信号分组压缩模块

    专门针对HVAC系统的信号特点，按照信号类型分组进行压缩
    """

    def __init__(self,
                 input_channels,
                 feature_dim,
                 hvac_groups=None,
                 feature_columns=None,
                 group_compression_ratios=None):
        super().__init__()

        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.hvac_groups = hvac_groups or self._get_default_hvac_groups()
        self.feature_columns = feature_columns

        # 默认压缩比例
        if group_compression_ratios is None:
            group_compression_ratios = {
                'temperature': 0.6,  # 温度信号通常有冗余
                'pressure': 0.7,     # 压力信号
                'flow': 0.8,         # 流量信号
                'valve': 0.5,        # 阀门状态信号冗余较多
                'fan': 0.7,          # 风机信号
                'other': 0.8         # 其他信号
            }

        self.group_compression_ratios = group_compression_ratios

        # 创建通道到组的映射
        self.channel_to_group = self._create_channel_mapping()

        # 为每个组创建压缩模块
        self.group_compressors = nn.ModuleDict()
        self.group_info = {}

        for group_name, signals in self.hvac_groups.items():
            group_channels = [
                ch for ch, g in self.channel_to_group.items() if g == group_name]
            if group_channels:
                input_ch = len(group_channels)
                compression_ratio = self.group_compression_ratios.get(
                    group_name, 0.8)
                output_ch = max(1, int(input_ch * compression_ratio))

                # 如果输出通道数为0，跳过该组
                if output_ch == 0:
                    continue

                self.group_compressors[group_name] = ChannelCompressionModule(
                    input_ch, output_ch, feature_dim,
                    compression_strategy='attention_guided'
                )

                self.group_info[group_name] = {
                    'channels': group_channels,
                    'input_size': input_ch,
                    'output_size': output_ch,
                    'compression_ratio': compression_ratio
                }

        # 处理未分组的通道
        ungrouped_channels = [
            ch for ch, g in self.channel_to_group.items() if g == 'other']
        if ungrouped_channels:
            input_ch = len(ungrouped_channels)
            compression_ratio = self.group_compression_ratios.get('other', 0.8)
            output_ch = max(1, int(input_ch * compression_ratio))

            self.group_compressors['other'] = ChannelCompressionModule(
                input_ch, output_ch, feature_dim,
                compression_strategy='attention_guided'
            )

            self.group_info['other'] = {
                'channels': ungrouped_channels,
                'input_size': input_ch,
                'output_size': output_ch,
                'compression_ratio': compression_ratio
            }

        # 计算总的输出通道数
        self.total_output_channels = sum(
            info['output_size'] for info in self.group_info.values())

        print(f"HVAC信号分组压缩模块初始化:")
        print(f"  - 输入通道数: {input_channels}")
        print(f"  - 输出通道数: {self.total_output_channels}")
        print(f"  - 总体压缩比例: {self.total_output_channels / input_channels:.2f}")

        for group_name, info in self.group_info.items():
            print(f"  - {group_name}组: {info['input_size']} -> {info['output_size']} "
                  f"(压缩比例: {info['compression_ratio']:.2f})")

    def _get_default_hvac_groups(self):
        """获取默认的HVAC信号分组"""
        return {
            'temperature': ['TEMP', 'T_', 'TEMPERATURE'],
            'pressure': ['PRESS', 'P_', 'PRESSURE'],
            'flow': ['FLOW', 'F_', 'FLOW_RATE'],
            'valve': ['VALVE', 'V_', 'VLV'],
            'fan': ['FAN', 'BLOWER', 'MOTOR'],
        }

    def _create_channel_mapping(self):
        """创建通道索引到组的映射"""
        channel_to_group = {}

        if not self.feature_columns:
            # 如果没有特征列信息，均匀分配到各组
            for i in range(self.input_channels):
                group_idx = i % len(self.hvac_groups)
                group_name = list(self.hvac_groups.keys())[group_idx]
                channel_to_group[i] = group_name
            return channel_to_group

        # 为每个通道找到对应的组
        for channel_idx, column_name in enumerate(self.feature_columns):
            group_found = False

            for group_name, group_signals in self.hvac_groups.items():
                if any(signal.upper() in column_name.upper() for signal in group_signals):
                    channel_to_group[channel_idx] = group_name
                    group_found = True
                    break

            if not group_found:
                channel_to_group[channel_idx] = 'other'

        return channel_to_group

    def forward(self, x):
        """前向传播

        Args:
            x: [B, C, feature_dim] 输入特征

        Returns:
            compressed: [B, compressed_C, feature_dim] 分组压缩后的特征
        """
        B, C, D = x.shape

        compressed_groups = []

        # 按组处理
        for group_name, compressor in self.group_compressors.items():
            group_channels = self.group_info[group_name]['channels']

            if group_channels:
                # 提取该组的通道
                group_x = x[:, group_channels, :]  # [B, group_size, D]

                # 压缩
                # [B, compressed_group_size, D]
                compressed_group = compressor(group_x)
                compressed_groups.append(compressed_group)

        # 拼接所有组的压缩结果
        # [B, total_compressed_C, D]
        compressed = torch.cat(compressed_groups, dim=1)

        return compressed
