import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class VGG(nn.Module):
    def __init__(self, features, num_classes=10, in_channels=3):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# VGG配置
cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name='VGG16', num_classes=10, in_channels=3, batch_norm=True):
    cfg = cfgs[model_name]
    features = make_layers(cfg, in_channels=in_channels, batch_norm=batch_norm)
    return VGG(features, num_classes=num_classes, in_channels=in_channels)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.model = vgg(
            model_name=getattr(configs, 'vgg_type', 'VGG19'),
            num_classes=configs.num_class,
            in_channels=configs.enc_in,
            batch_norm=True
        )
    def forward(self, x_enc):
        dec_out = self.model(x_enc)
        return dec_out  # [B, N] 