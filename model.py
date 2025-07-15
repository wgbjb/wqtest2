import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import Swin_T_Weights
from torchvision.ops import FeaturePyramidNetwork
import torch.nn.functional as F

class SwinBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化Swin Transformer
        self.swin = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        # 获取Swin Transformer的各个阶段
        self.stages = nn.ModuleList([])
        for i, layer in enumerate(self.swin.features):
            self.stages.append(layer)
    
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.stages):
            x = layer(x)
            # 保存每个阶段的输出
            if i in [1, 3, 5, 7]:  # 这些索引对应不同阶段的输出
                features.append(x)
        
        # 打印每个特征图的形状以进行调试
        for i, feat in enumerate(features):
            print(f"Feature {i} shape: {feat.shape}")
        
        return features

class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        self.fuse_conv = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=1)

    def forward(self, features):
        # 检查特征数量是否匹配
        assert len(features) == len(self.in_channels_list), \
            f"Expected {len(self.in_channels_list)} features, got {len(features)}"
        
        # 检查并确保每个特征是NCHW格式
        for i, (f, c) in enumerate(zip(features, self.in_channels_list)):
            if f.shape[1] != c:  # 如果通道维度不匹配
                print(f"Feature {i} shape: {f.shape}, expected channels: {c}")
                if f.shape[-1] == c:  # 如果最后一维是通道数
                    features[i] = f.permute(0, 3, 1, 2)  # NHWC -> NCHW
                    print(f"Permuted feature {i} shape: {features[i].shape}")
        
        # 应用1x1卷积调整通道数
        p = [self.lateral_convs[i](f) for i, f in enumerate(features)]
        
        # 上采样融合
        for i in range(len(p)-2, -1, -1):
            p[i] = p[i] + F.interpolate(p[i+1], size=p[i].shape[2:], mode='nearest')
        
        # 下采样融合
        for i in range(1, len(p)):
            p[i] = p[i] + F.interpolate(p[i-1], size=p[i].shape[2:], mode='nearest')
        
        # 合并所有特征
        fused = torch.cat([F.interpolate(f, size=p[0].shape[2:], mode='nearest') for f in p], dim=1)
        return self.fuse_conv(fused)

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Swin-T的各层通道数
        self.backbone = SwinBackbone()
        self.decoder = BiFPN([96, 192, 384, 768])
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        fused = self.decoder(features)
        depth = self.final_conv(fused)
        return depth