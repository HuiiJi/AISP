import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from thop import profile

# 注意力 Attention
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))  # nn.Parameter() 将该tensor变为可学习的参数

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=(1,1), bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=(3,3), padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=(1,1), bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        q, k, v = self.qkv_conv(qkv).chunk(3, dim=1)  # 在1维度上拆成三份

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).reshape(b, -1, h, w)
        out = self.project_out(out)   # 通道层MLP
        return out


# FeedForward
class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=(1,1), bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=(3,3), padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=(1,1), bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.conv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mu = x.mean(axis=1, keepdims=True)
        sigma = x.var(axis=1, keepdims=True)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        # 通道间归一化, 之后注意力, 之后残差连接
        x = x + self.attn(self.norm1(x))
        # 卷积, 残差
        x = x + self.ffn(self.norm2(x))
        return x


# 降采样, 通道数加倍, 特征图大小/4
class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=(3,3), padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


# 上采样, 通道数减半, 特征图大小*4
class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(self, in_channels=4, out_channels=4,
                 num_blocks=[4, 6, 6, 8],      # num_blocks=[4, 6, 6, 8]
                 num_heads=[1, 2, 4, 8],       # num_heads=[1, 2, 4, 8]
                 channels=[16, 32, 64, 128],  # channels=[48, 96, 192, 384]
                 num_refinement=4,
                 expansion_factor=2.66):
        super(Restormer, self).__init__()

        # 预卷积
        self.embed_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=(3, 3), padding=1),
            # nn.Conv2d(channels[0], channels[0], kernel_size=(5, 5), padding=2),
            # nn.Conv2d(channels[0], channels[0], kernel_size=(1, 1), padding=0),
        )

        # ---------------- down ------------------
        # 创建 encorder 的 transformer 的 block 块
        self.encoders = nn.ModuleList(
            [nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) \
             for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)]
        )
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList(
            [DownSample(num_ch) for num_ch in channels[:-1]]
        )
        # ---------------- up --------------------
        self.ups = nn.ModuleList(
            [UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]]
        )
        # 通道数调整, concat之后的特征图通道数调整
        self.reduces = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i - 1], kernel_size=(1,1), bias=False) for i in reversed(range(2, len(channels)))]
        )
        # 创建 decorder 的 transformer 的 block 块
        self.decoders = nn.ModuleList(
            [nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor) for _ in range(num_blocks[2])])]
        )
        self.decoders.append(
            nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor) for _ in range(num_blocks[1])])
        )
        # the channel of last one is not change
        self.decoders.append(
            nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_blocks[0])])
        )

        self.refinement = nn.Sequential(
            *[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_refinement)]
        )

        # 输出卷积
        self.output = nn.Sequential(
            nn.Conv2d(channels[1], out_channels, kernel_size=(3,3), padding=1),
            # nn.Conv2d(channels[1], channels[1], kernel_size=(1,1), padding=0),
            # nn.Conv2d(channels[1], out_channels, kernel_size=(5,5), padding=0),
        )


    def forward_features(self, inputs: torch.Tensor):
        n, c, h, w = inputs.shape
        h_pad = 32 - h % 32 
        w_pad = 32 - w % 32 
        inputs = F.pad(inputs, (0, w_pad, 0, h_pad), 'constant')
        return inputs
    
    
    def forward(self, inputs: torch.Tensor):
        n, c, h, w = inputs.shape
        inp = self.forward_features(inputs)

        fo = self.embed_conv(inp)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr)
        out = out[:, :, :h, :w]
        return out


if __name__ == '__main__':
    flops, params = profile(Restormer(), inputs=(torch.randn(1, 4, 64, 64),))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G' )
    print('Params = ' + str(params / 1000 ** 2) + 'M')
pass