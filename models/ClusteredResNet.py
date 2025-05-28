import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RestNet import ResNet, BasicBlock
import copy
from typing import Dict, List
class ClusteredResNet(nn.Module):
    def __init__(self, configs):
        super(ClusteredResNet, self).__init__()
        self.configs = configs
        self.in_channels = configs.enc_in
        
        # 从配置中获取通道分组
        self.channel_groups = configs.channel_groups  # 例如: [[0,1,2], [3,4,5], [6,7,8]]
        self.num_groups = len(self.channel_groups)
        
        # 每个分组的卷积层 - 减少通道数
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(len(group), 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ) for group in self.channel_groups
        ])
        
        # 使用ResNet作为合并后的网络
        self.merge_net = ResNet(
            block=BasicBlock,
            layers=[3, 3, 3, 3],  # ResNet18的层数配置
            num_classes=configs.num_class,
            in_channels=16 * self.num_groups  # 输入通道数为所有分组输出通道数的总和
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
    
    def forward(self, x):
        # 输入形状: [B, H, W, C]
        batch_size = x.size(0)
        
        # 调整输入维度顺序 [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # 对每个分组分别进行卷积
        group_features = []
        for i, group in enumerate(self.channel_groups):
            # 提取当前分组的通道
            group_channels = x[:, group, :, :]
            # 对当前分组进行卷积
            conv_out = self.group_convs[i](group_channels)
            group_features.append(conv_out)
        
        # 合并所有分组的特征
        x = torch.cat(group_features, dim=1)
        
        # 使用ResNet进行最终处理 [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2,3,1)
        x = self.merge_net(x)
        
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        
        # 如果没有指定通道分组，则使用默认分组
        if not hasattr(configs, 'channel_groups'):
            # 默认将通道平均分组
            num_channels = configs.enc_in
            channels_per_group = num_channels // 3  # 默认分成3组
            configs.channel_groups = [
                list(range(i, i + channels_per_group)) 
                for i in range(0, num_channels, channels_per_group)
            ]
            # 处理余数
            if num_channels % 3 != 0:
                configs.channel_groups[-1].extend(range(num_channels - (num_channels % 3), num_channels))
        
        self.model = ClusteredResNet(configs)
        
    def forward(self, x_enc):
        dec_out = self.model(x_enc)
        return dec_out  # [B, N] 


def _ensure_non_zero(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.cpu()
    if mask.sum().item() == 0:
        mask[0] = True
    return mask

def _prune_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d, 
                  in_idx: List[int] = None, 
                  out_mask: torch.Tensor = None) -> (nn.Conv2d, nn.BatchNorm2d):
    # 剪枝输出通道
    if out_mask is not None:
        out_mask = out_mask.cpu()
        out_mask = _ensure_non_zero(out_mask)
        idx_out = out_mask.nonzero(as_tuple=False).squeeze(1).tolist()
        # 确保索引在有效范围内
        idx_out = [i for i in idx_out if i < bn.weight.size(0)]
        if not idx_out:  # 如果所有索引都无效，保留至少一个通道
            idx_out = [0]
        new_out_channels = len(idx_out)
        # 剪枝输入通道
        if in_idx is not None:
            # 确保输入索引在有效范围内
            in_idx = [i for i in in_idx if i < conv.in_channels]
            if not in_idx:  # 如果所有索引都无效，保留至少一个通道
                in_idx = [0]
            new_in_channels = len(in_idx)
            new_conv = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=new_out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None
            )
            new_conv.weight.data = conv.weight.data[idx_out][:, in_idx, :, :].clone()
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data[idx_out].clone()
        else:
            new_conv = nn.Conv2d(
                in_channels=conv.in_channels,
                out_channels=new_out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None
            )
            new_conv.weight.data = conv.weight.data[idx_out].clone()
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data[idx_out].clone()
    else:
        new_conv = copy.deepcopy(conv)
        idx_out = list(range(conv.out_channels))
    
    # 剪枝BN层
    new_bn = nn.BatchNorm2d(new_conv.out_channels)
    if out_mask is not None:
        # 确保所有索引都在有效范围内
        valid_idx_out = [i for i in idx_out if i < bn.weight.size(0)]
        if not valid_idx_out:
            valid_idx_out = [0]
        new_bn.weight.data = bn.weight.data[valid_idx_out].clone()
        new_bn.bias.data = bn.bias.data[valid_idx_out].clone()
        new_bn.running_mean = bn.running_mean[valid_idx_out].clone()
        new_bn.running_var = bn.running_var[valid_idx_out].clone()
    else:
        new_bn.load_state_dict(bn.state_dict())
    return new_conv, new_bn

def prune_resnet_block(block: BasicBlock, in_channels: int, global_threshold: float) -> (BasicBlock, int):
    # 计算当前块的输出通道数
    out_channels = block.conv2.out_channels
    
    # 对第一个卷积层进行剪枝
    mask1 = (block.bn1.weight.data.abs() > global_threshold).cpu()
    mask1 = _ensure_non_zero(mask1)
    idx1 = mask1.nonzero(as_tuple=False).squeeze(1).tolist()
    new_conv1, new_bn1 = _prune_conv_bn(
        block.conv1, 
        block.bn1, 
        in_idx=list(range(in_channels)),
        out_mask=mask1
    )
    
    # 对第二个卷积层进行剪枝
    mask2 = (block.bn2.weight.data.abs() > global_threshold).cpu()
    mask2 = _ensure_non_zero(mask2)
    idx2 = mask2.nonzero(as_tuple=False).squeeze(1).tolist()
    new_conv2, new_bn2 = _prune_conv_bn(
        block.conv2, 
        block.bn2,
        in_idx=idx1,
        out_mask=mask2
    )
    
    # 处理downsample层
    new_downsample = None
    if block.downsample is not None:
        ds_conv, ds_bn = block.downsample[0], block.downsample[1]
        # 确保downsample的输出通道数与主路径匹配
        new_ds_conv, new_ds_bn = _prune_conv_bn(
            ds_conv, 
            ds_bn,
            in_idx=list(range(in_channels)),
            out_mask=mask2  # 使用与主路径相同的mask
        )
        new_downsample = nn.Sequential(new_ds_conv, new_ds_bn)
    elif block.stride != 1 or in_channels != len(idx2):
        # 如果没有downsample但需要改变通道数或步长，添加downsample
        new_ds_conv = nn.Conv2d(
            in_channels,
            len(idx2),
            kernel_size=1,
            stride=block.stride,
            bias=False
        ).to(block.conv1.weight.device)
        new_ds_bn = nn.BatchNorm2d(len(idx2)).to(block.conv1.weight.device)
        new_downsample = nn.Sequential(new_ds_conv, new_ds_bn)
    
    # 创建新的BasicBlock
    new_block = BasicBlock(
        in_channels=in_channels,
        out_channels=len(idx2),
        stride=block.stride,
        downsample=new_downsample
    )
    new_block.conv1 = new_conv1
    new_block.bn1 = new_bn1
    new_block.conv2 = new_conv2
    new_block.bn2 = new_bn2
    
    return new_block, len(idx2)

def global_prune(model, prune_ratio: float, device=None):
    # 获取原始模型的设备
    original_device = next(model.parameters()).device
    model = copy.deepcopy(model)
    all_gammas = []
    for group in model.model.group_convs:
        all_gammas.append(group[1].weight.data.abs().clone())
        all_gammas.append(group[4].weight.data.abs().clone())
    for name, module in model.model.merge_net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            all_gammas.append(module.weight.data.abs().clone())
    global_threshold = torch.quantile(torch.cat(all_gammas), prune_ratio)
    total_output_channels = 0
    new_group_convs = nn.ModuleList()
    group_out_indices = []
    
    # 处理分组卷积
    for group in reversed(model.model.group_convs):
        conv1, bn1, relu1, conv2, bn2, relu2 = group
        mask1 = (bn1.weight.data.abs() > global_threshold).cpu()
        mask1 = _ensure_non_zero(mask1)
        idx1 = mask1.nonzero(as_tuple=False).squeeze(1).tolist()
        new_conv1, new_bn1 = _prune_conv_bn(conv1, bn1, out_mask=mask1)
        mask2 = (bn2.weight.data.abs() > global_threshold).cpu()
        mask2 = _ensure_non_zero(mask2)
        idx2 = mask2.nonzero(as_tuple=False).squeeze(1).tolist()
        new_conv2, new_bn2 = _prune_conv_bn(
            conv2, bn2, 
            in_idx=idx1,
            out_mask=mask2
        )
        total_output_channels += new_conv2.out_channels
        group_out_indices.insert(0, list(range(total_output_channels - new_conv2.out_channels, total_output_channels)))
        new_group = nn.Sequential(
            new_conv1, new_bn1, relu1,
            new_conv2, new_bn2, relu2
        )
        new_group_convs.insert(0, new_group)
    model.model.group_convs = new_group_convs
    
    # 更新merge_net的输入通道数
    merge_net = model.model.merge_net
    merge_net.conv1 = nn.Conv2d(
        total_output_channels,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    ).to(original_device)
    
    # 处理ResNet的各个层
    current_channels = 64  # 从conv1的输出通道数开始
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(merge_net, layer_name)
        new_blocks = []
        for block in layer:
            new_block, out_channels = prune_resnet_block(block, current_channels, global_threshold)
            new_blocks.append(new_block)
            current_channels = out_channels
        setattr(merge_net, layer_name, nn.Sequential(*new_blocks))
    
    # 更新最后的全连接层
    merge_net.fc = nn.Linear(current_channels, merge_net.fc.out_features).to(original_device)
    
    # 确保模型在正确的设备上
    model = model.to(original_device)
    
    return model
    
 