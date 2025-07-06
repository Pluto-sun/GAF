# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失函数
    
    标签平滑通过将硬标签（one-hot）转换为软标签来防止过拟合和过度自信，
    特别适用于类别间相似性较高的情况。
    
    Args:
        smoothing (float): 平滑因子，通常在0.05-0.2之间
        num_classes (int): 类别数量
        dim (int): softmax的维度，默认为-1
        weight (Tensor): 各类别的权重，用于处理类别不平衡
        
    公式：
        软标签 = (1 - smoothing) * 硬标签 + smoothing / num_classes
    """
    def __init__(self, smoothing=0.1, num_classes=None, dim=-1, weight=None):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.weight = weight
        
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] 模型预测logits
            target: [N] 真实标签
        """
        if self.num_classes is None:
            self.num_classes = pred.size(1)
            
        pred = pred.log_softmax(dim=self.dim)
        
        # 创建软标签
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        # 如果有类别权重，应用权重
        if self.weight is not None:
            # 将权重扩展到batch维度
            weight_expanded = self.weight[target]  # [N]
            loss = torch.sum(-true_dist * pred, dim=self.dim) * weight_expanded
            return loss.mean()
        else:
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class FocalLoss(nn.Module):
    """
    Focal Loss 专门处理难分类样本和类别不平衡问题
    
    通过动态调整损失权重，让模型更专注于难分类的样本，
    减少易分类样本对损失的贡献。
    
    Args:
        alpha (float or Tensor): 平衡因子，用于处理类别不平衡
        gamma (float): 聚焦参数，gamma=0时退化为交叉熵
        weight (Tensor): 各类别权重
        reduction (str): 'mean', 'sum', 'none'
        
    公式：
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] 模型预测logits
            target: [N] 真实标签
        """
        ce_loss = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 计算alpha_t
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha[target]
            
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    类别平衡损失函数
    
    基于每个类别的有效样本数来重新加权损失，
    适用于长尾分布的数据集。
    
    Args:
        samples_per_class (list): 每个类别的样本数
        beta (float): 重采样超参数，通常在0.9-0.99之间
        loss_type (str): 基准损失类型，'focal', 'ce', 'sigmoid'
        gamma (float): 如果使用focal loss的gamma参数
    """
    def __init__(self, samples_per_class, beta=0.9999, loss_type='ce', gamma=2.0):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        self.weights = torch.tensor(weights / np.sum(weights) * len(weights), dtype=torch.float32)
        self.loss_type = loss_type
        self.gamma = gamma
        
    def forward(self, pred, target):
        weights = self.weights.to(pred.device)
        
        if self.loss_type == 'ce':
            return F.cross_entropy(pred, target, weight=weights)
        elif self.loss_type == 'focal':
            ce_loss = F.cross_entropy(pred, target, weight=weights, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss (Additive Angular Margin Loss)
    
    通过在角度空间增加边界来增强类别间的区分度，
    特别适用于特征相似的类别。
    
    Args:
        in_features (int): 输入特征维度
        out_features (int): 输出类别数
        s (float): 特征缩放因子
        m (float): 角度边界
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, features, target):
        # 特征和权重归一化
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # 计算角度边界
        phi = cosine * torch.cos(torch.tensor(self.m)) - sine * torch.sin(torch.tensor(self.m))
        
        # 只对真实标签应用边界
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return F.cross_entropy(output, target)


class CenterLoss(nn.Module):
    """
    Center Loss 中心损失
    
    通过最小化同类特征到类中心的距离来增强类内聚合度，
    同时配合交叉熵损失使用。
    
    Args:
        num_classes (int): 类别数量
        feat_dim (int): 特征维度
        alpha (float): 中心更新的学习率
    """
    def __init__(self, num_classes, feat_dim, alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, features, target):
        """
        Args:
            features: [N, feat_dim] 提取的特征
            target: [N] 真实标签
        """
        batch_size = features.size(0)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long().to(features.device)
        target_expanded = target.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = target_expanded.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        # 更新中心
        if self.training:
            with torch.no_grad():
                for i, label in enumerate(target):
                    self.centers[label] = self.centers[label] * (1 - self.alpha) + features[i] * self.alpha
                    
        return loss


class ConfidencePenaltyLoss(nn.Module):
    """
    置信度惩罚损失
    
    惩罚过度自信的预测，鼓励模型输出更平衡的概率分布，
    有助于缓解类别间相似性问题。
    
    Args:
        beta (float): 惩罚强度
    """
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta
        
    def forward(self, pred, target):
        # 基础交叉熵损失
        ce_loss = F.cross_entropy(pred, target)
        
        # 置信度惩罚：KL散度相对于均匀分布
        prob = F.softmax(pred, dim=1)
        log_prob = F.log_softmax(pred, dim=1)
        uniform_dist = torch.ones_like(prob) / prob.size(1)
        
        # KL(uniform || pred) = -H(uniform) + H_cross(uniform, pred)
        kl_penalty = F.kl_div(log_prob, uniform_dist, reduction='batchmean')
        
        return ce_loss - self.beta * kl_penalty


class TripletMarginLoss(nn.Module):
    """
    三元组边界损失
    
    通过三元组（锚点，正样本，负样本）学习，
    拉近同类样本，推远不同类样本。
    
    Args:
        margin (float): 边界值
        p (int): 距离度量的p值（1或2）
    """
    def __init__(self, margin=1.0, p=2):
        super().__init__()
        self.margin = margin
        self.p = p
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=p)
        
    def forward(self, features, target):
        """
        Args:
            features: [N, feat_dim] 提取的特征
            target: [N] 真实标签
        """
        # 简化版本：随机选择三元组
        batch_size = features.size(0)
        triplet_losses = []
        
        for i in range(batch_size):
            anchor = features[i:i+1]
            anchor_label = target[i]
            
            # 找正样本（同类别）
            pos_mask = (target == anchor_label) & (torch.arange(batch_size) != i)
            if pos_mask.sum() > 0:
                pos_idx = torch.where(pos_mask)[0][0]
                positive = features[pos_idx:pos_idx+1]
                
                # 找负样本（不同类别）
                neg_mask = (target != anchor_label)
                if neg_mask.sum() > 0:
                    neg_idx = torch.where(neg_mask)[0][0]
                    negative = features[neg_idx:neg_idx+1]
                    
                    loss = self.triplet_loss(anchor, positive, negative)
                    triplet_losses.append(loss)
        
        if triplet_losses:
            return torch.stack(triplet_losses).mean()
        else:
            return torch.tensor(0.0, requires_grad=True, device=features.device)


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    将多个损失函数按权重组合，例如：
    - CrossEntropy + CenterLoss
    - LabelSmoothing + FocalLoss
    - CrossEntropy + ConfidencePenalty
    
    Args:
        losses (dict): 损失函数字典，格式为 {'loss_name': (loss_fn, weight)}
    """
    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        
        for name, (loss_fn, weight) in losses.items():
            self.losses[name] = loss_fn
            self.weights[name] = weight
            
    def forward(self, *args, **kwargs):
        total_loss = 0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            loss = loss_fn(*args, **kwargs)
            weighted_loss = self.weights[name] * loss
            total_loss += weighted_loss
            loss_dict[name] = loss.item()
            
        return total_loss, loss_dict


def get_loss_function(loss_type, num_classes, **kwargs):
    """
    损失函数工厂函数
    
    Args:
        loss_type (str): 损失函数类型
        num_classes (int): 类别数量
        **kwargs: 其他参数
    
    Returns:
        loss_fn: 损失函数实例
    """
    if loss_type == 'ce':
        weight = kwargs.get('class_weights', None)
        return nn.CrossEntropyLoss(weight=weight)
    
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        weight = kwargs.get('class_weights', None)
        return LabelSmoothingCrossEntropy(smoothing=smoothing, num_classes=num_classes, weight=weight)
    
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 1.0)
        gamma = kwargs.get('gamma', 2.0)
        weight = kwargs.get('class_weights', None)
        return FocalLoss(alpha=alpha, gamma=gamma, weight=weight)
    
    elif loss_type == 'class_balanced':
        samples_per_class = kwargs.get('samples_per_class', None)
        beta = kwargs.get('beta', 0.9999)
        base_loss = kwargs.get('base_loss', 'ce')
        gamma = kwargs.get('gamma', 2.0)
        if samples_per_class is None:
            raise ValueError("samples_per_class is required for class_balanced loss")
        return ClassBalancedLoss(samples_per_class, beta=beta, loss_type=base_loss, gamma=gamma)
    
    elif loss_type == 'confidence_penalty':
        beta = kwargs.get('beta', 0.1)
        return ConfidencePenaltyLoss(beta=beta)
    
    elif loss_type == 'combined':
        # 示例组合：标签平滑 + 置信度惩罚
        smoothing = kwargs.get('smoothing', 0.1)
        penalty_beta = kwargs.get('penalty_beta', 0.05)
        
        label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=smoothing, num_classes=num_classes)
        penalty_loss = ConfidencePenaltyLoss(beta=penalty_beta)
        
        losses = {
            'label_smoothing': (label_smooth_loss, 1.0),
            'confidence_penalty': (penalty_loss, 0.5)
        }
        return CombinedLoss(losses)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# 预定义的损失配置方案
LOSS_CONFIGS = {
    'hvac_similar_classes': {
        'type': 'label_smoothing',
        'smoothing': 0.15,  # 较高的平滑因子用于相似类别
        'description': 'HVAC异常检测中相似类别的推荐配置'
    },
    
    'imbalanced_classes': {
        'type': 'focal',
        'alpha': 1.0,
        'gamma': 2.0,
        'description': '类别不平衡问题的推荐配置'
    },
    
    'hard_samples': {
        'type': 'focal',
        'alpha': 0.25,  # 降低易分类样本权重
        'gamma': 3.0,   # 增强难样本聚焦
        'description': '难分类样本的推荐配置'
    },
    
    'overconfidence_prevention': {
        'type': 'confidence_penalty',
        'beta': 0.1,
        'description': '防止过度自信的推荐配置'
    },
    
    'comprehensive_solution': {
        'type': 'combined',
        'smoothing': 0.1,
        'penalty_beta': 0.05,
        'description': '综合解决方案：标签平滑 + 置信度惩罚'
    }
}


def get_recommended_loss_config(problem_type):
    """
    获取推荐的损失配置
    
    Args:
        problem_type (str): 问题类型
        
    Returns:
        dict: 损失配置
    """
    return LOSS_CONFIGS.get(problem_type, LOSS_CONFIGS['hvac_similar_classes'])
