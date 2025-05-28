import torch
import torch.nn as nn
import copy

# 定义剪枝函数
def prune_model(model, prune_ratio, device=None):
    # 只对group_convs做剪枝
    group_convs = model.model.group_convs  # 适配你的ClusteredResNet结构
    masks = []
    for seq in group_convs:
        # 只剪第一个BN
        bn1 = seq[1]
        gamma = bn1.weight.data.abs()
        threshold = torch.quantile(gamma, prune_ratio)
        mask = gamma > threshold
        masks.append(mask)
    # 重建group_convs
    new_group_convs = prune_group_convs(group_convs, masks)
    model.model.group_convs = new_group_convs
    if device is not None:
        model = model.to(device)
    return model

# 定义带L1正则化的损失函数
def loss_with_regularization(model, outputs, targets, lambda_l1=0.001):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    l1_reg = 0.0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            l1_reg += torch.norm(module.weight, p=1)
    loss += lambda_l1 * l1_reg
    return loss

def prune_conv_bn(conv, bn, mask_out):
    # mask_out: 1D bool tensor, shape = [out_channels]
    idx_out = mask_out.nonzero(as_tuple=False).squeeze(1)
    # 新建Conv2d
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=len(idx_out),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode
    )
    # 迁移权重
    new_conv.weight.data = conv.weight.data[idx_out].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[idx_out].clone()
    # 新建BN
    new_bn = nn.BatchNorm2d(len(idx_out))
    new_bn.weight.data = bn.weight.data[idx_out].clone()
    new_bn.bias.data = bn.bias.data[idx_out].clone()
    new_bn.running_mean = bn.running_mean[idx_out].clone()
    new_bn.running_var = bn.running_var[idx_out].clone()
    return new_conv, new_bn, idx_out

def prune_group_convs(group_convs, masks):
    # group_convs: nn.ModuleList of nn.Sequential
    # masks: list of 1D bool tensors, one for each group
    new_group_convs = nn.ModuleList()
    for seq, mask in zip(group_convs, masks):
        # 假设结构: Conv2d -> BN2d -> ReLU -> Conv2d -> BN2d -> ReLU
        conv1, bn1, relu1, conv2, bn2, relu2 = seq
        # 剪第一个BN/Conv
        conv1_new, bn1_new, idx1 = prune_conv_bn(conv1, bn1, mask)
        # 剪第二个BN/Conv（输入通道也要同步）
        # 这里假设第二个Conv的in_channels等于第一个BN的输出
        conv2_new = nn.Conv2d(
            in_channels=len(idx1),
            out_channels=conv2.out_channels,
            kernel_size=conv2.kernel_size,
            stride=conv2.stride,
            padding=conv2.padding,
            dilation=conv2.dilation,
            groups=conv2.groups,
            bias=(conv2.bias is not None),
            padding_mode=conv2.padding_mode
        )
        conv2_new.weight.data = conv2.weight.data[:, idx1, :, :].clone()
        if conv2.bias is not None:
            conv2_new.bias.data = conv2.bias.data.clone()
        # BN2不剪
        new_seq = nn.Sequential(conv1_new, bn1_new, relu1, conv2_new, bn2, relu2)
        new_group_convs.append(new_seq)
    return new_group_convs