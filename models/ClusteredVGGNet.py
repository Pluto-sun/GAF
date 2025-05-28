import torch
import torch.nn as nn
import torch.nn.functional as F
from models.VGGNet import vgg


def parse_channel_groups(channel_groups):
    if isinstance(channel_groups, str):
        groups = []
        for group in channel_groups.split('|'):
            start, end = map(int, group.split(','))
            groups.append(list(range(start, end + 1)))
        return groups
    elif isinstance(channel_groups, list):
        return channel_groups
    else:
        raise ValueError("channel_groups must be str or list")

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x-1).bit_length()

class ClusteredVGGNet(nn.Module):
    def __init__(self, configs, merge_channels=64):
        super(ClusteredVGGNet, self).__init__()
        self.configs = configs
        self.in_channels = configs.enc_in
        self.channel_groups = configs.channel_groups  # 例如: [[0,1,2], [3,4], [5], ...]
        self.num_groups = len(self.channel_groups)
        self.group_out_channels = []
        self.group_convs = nn.ModuleList()

        # # 动态为每组构建卷积
        # for group in self.channel_groups:
        #     n = len(group)
        #     c_out = next_power_of_2(n)
        #     self.group_out_channels.append(c_out)
        #     self.group_convs.append(
        #         nn.Sequential(
        #             nn.Conv2d(n, 2*c_out, kernel_size=1),
        #             nn.BatchNorm2d(2*c_out),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(2*c_out, c_out, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(c_out),
        #             nn.ReLU(inplace=True)
        #         )
        #     )
        # total_out = sum(self.group_out_channels)
        # self.merge_net = nn.Sequential(
        #     nn.Conv2d(total_out, merge_channels, kernel_size=1),
        #     nn.BatchNorm2d(merge_channels),
        #     nn.ReLU(inplace=True)
        # )
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
        self.vgg = vgg(
            model_name=getattr(configs, 'vgg_type', 'VGG16'),
            num_classes=configs.num_class,
            in_channels=16 * self.num_groups,
            batch_norm=True
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        group_features = []
        for i, group in enumerate(self.channel_groups):
            group_channels = x[:, group, :, :]  # [B, n, H, W]
            conv_out = self.group_convs[i](group_channels)  # [B, c_out, H, W]
            group_features.append(conv_out)
        x = torch.cat(group_features, dim=1)  # [B, sum(c_out), H, W]
        # x = self.merge_net(x)  # [B, merge_channels, H, W]
        x=x.permute(0, 2,3,1)
        x = self.vgg(x)        # [B, num_classes]
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        if not hasattr(configs, 'channel_groups'):
            num_channels = configs.enc_in
            channels_per_group = num_channels // 3
            configs.channel_groups = [
                list(range(i, i + channels_per_group))
                for i in range(0, num_channels, channels_per_group)
            ]
            if num_channels % 3 != 0:
                configs.channel_groups[-1].extend(range(num_channels - (num_channels % 3), num_channels))
        # else:
        #     configs.channel_groups = parse_channel_groups(configs.channel_groups)
        self.model = ClusteredVGGNet(configs)
    def forward(self, x_enc):
        dec_out = self.model(x_enc)
        return dec_out  # [B, N]

