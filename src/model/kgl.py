import math

import torch
import torch.nn as nn


class KernelGaussianLayer(nn.Module):
    def __init__(self, in_channels=64, mu_threshold=0, sigma_threshold=0.1) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.empty(1, in_channels).normal_(std=2))
        self.sigma = nn.Parameter(torch.abs(torch.empty(1, in_channels).normal_(std=2)))
        self.mu_threshold = mu_threshold
        self.sigma_threshold = sigma_threshold

        self.in_channels = in_channels

    def _clamp_parameter(self):
        self.mu.data = self.mu.clamp(min=self.mu_threshold)
        self.sigma.data = self.sigma.clamp(min=self.sigma_threshold)

    def forward(self, x):
        # 裁剪参数
        self._clamp_parameter()
        
        # 获取输入的维度
        x_dim = x.ndim
        if x_dim == 2:
            # GAP + flatten 的输出或者 FC 层的输出 [b, c]
            norm_out = (x - self.mu) / self.sigma  # 归一化
        elif x_dim == 4:
            # 卷积层的输出 [b, c, h, w]
            norm_out = (x - self.mu.reshape(1, -1, 1, 1)) / self.sigma.reshape(1, -1, 1, 1)  # 归一化
            
        p_out = 0.5 * (1 + torch.erf(norm_out / math.sqrt(2)))  # 概率
        weight = torch.where(p_out < 0.5, p_out, 1 - p_out)

        return weight * x

    def extra_repr(self) -> str:
        return 'in_channels={in_channels}, mu_threshold={mu_threshold}, sigma_threshold={sigma_threshold}'.format(**self.__dict__)
