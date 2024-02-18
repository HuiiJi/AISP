# ------------------------------------------------------------------------
# Copyright (c) 2022 Murufeng. All Rights Reserved.
# ------------------------------------------------------------------------
'''
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


'''
ref. 
@article{chu2021tlsc,
  title={Revisiting Global Statistics Aggregation for Improving Image Restoration},
  author={Chu, Xiaojie and Chen, Liangyu and and Chen, Chengpeng and Lu, Xin},
  journal={arXiv preprint arXiv:2112.04491},
  year={2021}
}
'''
class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

class LayerNormFunction(torch.autograd.Function):
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm(nn.Module):
    def __init__(self, c):
        super(LayerNorm, self).__init__()
        # if isinstance(normalized_shape, numbers.Integral):
        #     normalized_shape = (normalized_shape,)
        # normalized_shape = normalized_shape.shape

        # assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
 
        self.c = c

    def forward(self, x, eps=1e-6):
        # print(x.shape)
        # mu = x.mean(1, keepdim=True)
        # var = (x - mu).pow(2).mean(1, keepdim=True)
        # # sigma = x.var([2, 3], keepdim=True, unbiased=False)
        # # y = (x - mu) / torch.sqrt((var + eps))
        # y = (x - mu) / (var + eps).sqrt()

        # N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        # y = self.weight * y + self.bias
        # y = torch.reshape(self.weight, (1, self.c, 1, 1)) * y + torch.reshape(self.bias, (1, self.c, 1, 1))
        y = self.weight.view(1, self.c, 1, 1) * y + self.bias.view(1, self.c, 1, 1)
        
        # print('mu shape: ', mu.shape)
        # print('sigma shape: ', sigma.shape)
        # c = x.shape[1]
        # self.weight = paddle.reshape(self.weight, shape=[1, c, 1, 1])
        # self.bias = paddle.reshape(self.bias, shape=[1, c, 1, 1])
        # print('weight shape: ', self.weight.shape)
        # print('bias shape: ', self.bias.shape)
        #out = (x - mu) / paddle.sqrt(sigma) * self.weight + self.bias
        # print("out shape: ", out.shape)
        return y#(x - mu) / paddle.sqrt(sigma) * self.weight + self.bias


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

def layer_norm_process(feature: torch.Tensor, beta=0., gamma=1., eps=1e-5):
    var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
    # 均值
    mean = var_mean[1]
    # 方差
    var = var_mean[0]

    # layer norm process
    feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + eps)
    feature = feature * gamma + beta

    return feature

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # self.norm1 = LayerNorm2d(c)
        # self.norm2 = LayerNorm2d(c)
        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)
        # ###########################
        # self.norm1 = torch.nn.LayerNorm(c)
        # self.norm2 = torch.nn.LayerNorm(c)
        # ###########################
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        ##########################
        # x = torch.permute(x, (0, 3, 2, 1))
        ############################
        x = self.norm1(x)
        ##########################
        # x = torch.permute(x, (0, 3, 2, 1))
        #############################
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        ################################
        # yy = torch.permute(y, (0, 3, 2, 1))
       # yy =  layer_norm_process(y, eps=1e-5)
        # x = torch.permute(yy, (0, 3, 2, 1))
        #x = self.conv4(x)
        ####################################
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class NAFBlock2(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # self.norm1 = LayerNorm2d(c)
        # self.norm2 = LayerNorm2d(c)
        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)
        # ###########################
        # self.norm1 = torch.nn.LayerNorm(c)
        # self.norm2 = torch.nn.LayerNorm(c)
        # ###########################
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.scconv = ScConv(op_channel=c)

    def forward(self, inp):
        x = inp
        ##########################
        # x = torch.permute(x, (0, 3, 2, 1))
        ############################
        x = self.norm1(x)
        ##########################
        # x = torch.permute(x, (0, 3, 2, 1))
        #############################
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        ################################
        # yy = torch.permute(y, (0, 3, 2, 1))
       # yy =  layer_norm_process(y, eps=1e-5)
        # x = torch.permute(yy, (0, 3, 2, 1))
        #x = self.conv4(x)
        ####################################
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        x = y + x * self.gamma

        # return y + x * self.gamma
        return self.scconv(x)

class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, return_x = False):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        ori_inp = inp.clone()
        #x_start = inp.clone()    #+++++
        x = F.interpolate(inp, (512, 512), mode='bilinear')  #heiyanquan
        # x = F.interpolate(inp, (1024, 1024), mode='bilinear')  #body-fusetongyi

        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        # yanwei11 test
        # x = x * 1.2
        # x00  = x.clone()
        # x[(x < 0.05) & (x >= -0.05)] = 0
        x = F.interpolate(x, ori_inp.shape[2:], mode='bilinear')  # body-fusetongyi  推理的时候注意
        # x = F.interpolate(x, ori_inp.shape[2:], mode='bicubic')  # body-fusetongyi  推理的时候注意
        x_recon = x + ori_inp
        x_recon = x_recon[:, :3, :H, :W]
        '''
        feature = x[:, :3, :H, :W]   #+++++
        pred = feature + x_start   #+++++
        #pred = pred[:, :3, :H, :W]
        pred = pred.clamp(0.0, 1.0)     #+++++
        mask = x[:, 3, :H, :W]      #+++++
        mask = torch.unsqueeze(mask, 1)     #+++++
        #mask = torch.sigmoid(mask)      #+++++

        '''
        # if return_x:
        #     return x_recon, x
        return x_recon

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


class NAFNet_Upsample(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan // 2, 1, bias=False),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, return_x = False):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        ori_inp = inp.clone()
        #x_start = inp.clone()    #+++++
        x = F.interpolate(inp, (512, 512), mode='bilinear')  #heiyanquan
        # x = F.interpolate(inp, (1024, 1024), mode='bilinear')  #body-fusetongyi

        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        # yanwei11 test
        # x = x * 1.15

        # x00  = x.clone()
        # x[(x < 0.05) & (x >= -0.05)] = 0
        # import numpy as np
        # print(np.unique(x00.detach().cpu()))
        x = F.interpolate(x, ori_inp.shape[2:], mode='bilinear')  # body-fusetongyi  推理的时候注意
        # x = F.interpolate(x, ori_inp.shape[2:], mode='bicubic')  # body-fusetongyi  推理的时候注意
        x_r = x.clone()
        x_recon = x + ori_inp
        #x = x + ori_inp
        x_recon = x_recon[:, :3, :H, :W]
        '''
        feature = x[:, :3, :H, :W]   #+++++
        pred = feature + x_start   #+++++
        #pred = pred[:, :3, :H, :W]
        pred = pred.clamp(0.0, 1.0)     #+++++
        mask = x[:, 3, :H, :W]      #+++++
        mask = torch.unsqueeze(mask, 1)     #+++++
        #mask = torch.sigmoid(mask)      #+++++

        '''
        if return_x:
            return x_recon, x
        return x_recon

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetScConv(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock2(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock2(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock2(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        ori_inp = inp.clone()
        #x_start = inp.clone()    #+++++
        x = F.interpolate(inp, (512, 512), mode='bilinear')  #heiyanquan
        #x = F.interpolate(inp, (1024, 1024), mode='bilinear')  #body-fusetongyi

        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        # x00  = x.clone()
        # x[(x < 0.05) & (x >= -0.05)] = 0
        # import numpy as np
        # print(np.unique(x00.detach().cpu()))
        x = F.interpolate(x, ori_inp.shape[2:], mode='bilinear')  # body-fusetongyi

        x_r = x.clone()
        x = x + ori_inp
        #x = x + ori_inp
        x = x[:, :3, :H, :W]
        '''
        feature = x[:, :3, :H, :W]   #+++++
        pred = feature + x_start   #+++++
        #pred = pred[:, :3, :H, :W]
        pred = pred.clamp(0.0, 1.0)     #+++++
        mask = x[:, 3, :H, :W]      #+++++
        mask = torch.unsqueeze(mask, 1)     #+++++
        #mask = torch.sigmoid(mask)      #+++++

        '''
        return x
        #return x, x_r
        # return pred, mask       #+++++

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

####################SCConv##########################
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        #assert c_num    >= group_num
        self.group_num  = group_num
        self.gamma      = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.beta       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps

    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5 
                 ):
        super().__init__()
        
        self.gn = GroupBatchnorm2d( oup_channels, group_num = group_num )
        #self.gn = LayerNorm2d( oup_channels )
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = F.softmax(self.gn.gamma, dim=0)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        info_mask   = w_gamma > self.gate_treshold
        noninfo_mask= w_gamma <= self.gate_treshold
        x_1         = info_mask * reweigts * x
        x_2         = noninfo_mask * reweigts * x
        x           = self.reconstruct(x_1, x_2)
        return x
    
    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1, bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1, bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1, padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio, kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel, self.low_channel], dim=1)
        up,low  = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1, Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1, out2 = torch.split(out, out.size(1)//2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                op_channel:int,
                group_num:int = 16,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(op_channel, op_channel, kernel_size=1, stride=1, padding=0)
        self.SRU = SRU( op_channel, group_num = group_num,  gate_treshold = gate_treshold )
        self.CRU = CRU( op_channel, alpha = alpha, squeeze_radio = squeeze_radio , group_size = group_size , group_kernel_size = group_kernel_size )
        self.conv2 = nn.Conv2d(op_channel, op_channel, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.SRU(x)
        x = self.CRU(x)
        x = self.conv2(x)
        return x
#######################SCConv########################


class NAFNetCls(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 2)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        ori_inp = inp.clone()
  
        x = F.interpolate(inp, (512, 512), mode='bilinear')  #heiyanquan
        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        cls = self.bn(x)
        cls = self.avgpool(x.clone())
        cls = torch.flatten(cls, 1)
        cls = self.fc(cls)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        
        x = F.interpolate(x, ori_inp.shape[2:], mode='bilinear')
        x = x + ori_inp
        x = x[:, :3, :H, :W]
        return x, cls


    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetCls2(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 2)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width

        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock2(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock2(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock2(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        ori_inp = inp.clone()
        #x_start = inp.clone()    #+++++
        x = F.interpolate(inp, (512, 512), mode='bilinear')  #heiyanquan
        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        cls = self.bn(x)
        cls = self.avgpool(x.clone())
        cls = torch.flatten(cls, 1)
        cls = self.fc(cls)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        
        x = F.interpolate(x, ori_inp.shape[2:], mode='bilinear')
        x = x + ori_inp
        x = x[:, :3, :H, :W]
        return x, cls


    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



class NAFDiscriminator(nn.Module):
    # 512 
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        """定义判别器的网络结构"""
        super(NAFDiscriminator, self).__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        #self.discriminators = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        #_, _, H_ori, W_ori = inp.shape
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        feature = self.ending(x)
        x = feature + inp
        x = x[:, :3, :H, :W]
        return x
        
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class Discriminator0(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        """定义判别器的网络结构"""
        super(Discriminator, self).__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        #self.discriminators = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        self.dense = nn.Sequential(
            #nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        feature = self.ending(x)
        print('feature.shape:', feature.shape)
        feature = self.dense(feature)
        # x = feature + inp
        # x = x[:, :3, :H, :W]
        # return x
        return feature
        
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_AB):
        # Concatenate image and condition image by channels to produce input
        #img_input = torch.cat((img_A, img_B), 1)
        res = self.model(img_AB)
        return res


class NAFNet_ABPN(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x_ori = inp.clone()
        
        x = F.interpolate(inp, (512, 512), mode='bilinear')
        #x = F.interpolate(inp, (256, 256), mode='bilinear')
        #x_start = x.clone()
        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        x = F.interpolate(x, x_ori.shape[2:], mode='bilinear')
        x = x + x_ori
        x = x[:, :3, :H, :W]

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class NAFNetUnion(nn.Module):
    """NAFNetUnion implementation"""
    def __init__(self):
        super().__init__()
        self.deblur_block = NAFNet(img_channel=3, width=16, middle_blk_num=1, 
                                    enc_blk_nums=[1, 1, 1, 2], dec_blk_nums=[1, 1, 1, 1])
        self.delight_block = NAFNet(img_channel=3, width=8, middle_blk_num=1, 
                                    enc_blk_nums=[1, 1, 1, 2], dec_blk_nums=[1, 1, 1, 1])
        # maybe seg model
        # 1. seg input img
        # 2. get mask and inpaint
        # self.inpaint_block = NAFNet(img_channel=3, width=16, middle_blk_num=0, 
        #                             enc_blk_nums=[1, 0, 0, 1], dec_blk_nums=[1, 0, 0, 1])

    def forward(self, inp):
        """input"""
        # stage1: 去模糊模块
        deblur_out = self.deblur_block(inp)
        # stage2：去阴影模块
        delight_out = self.delight_block(deblur_out)
        # stage3：边缘处理模块
        # inpaint_out = self.inpaint_block(delight_out)
        # inpaint_mask = self.inpaint_block(delight_out)
        # inpaint_out = resize(inpaint_mask) * delight_out

        return deblur_out, delight_out, None #inpaint_out


if __name__ == "__main__":

    # model = NAFNetUnion()
    # model.eval()
    # print(len(model.state_dict().keys()))
    # for k, v in model.named_parameters():
    #     if 'enhance_block' not in k:
    #      v.requires_grad = False # freeze params

    # model.train()
    # for k, v in model.named_parameters():w
    #         print(k, v.requires_grad)

    # # print(model.state_dict().keys())
    # x = torch.randn([1, 3, 256, 256])
    # out = model(x)
    from torchsummary import summary
    TEACHER_MODEL = [{'width':8, 'middle_blk_num':1, 'enc_blk_nums':[1, 1, 1, 2], 'dec_blk_nums':[1, 1, 1, 1]}]
    teacher_model = NAFNet(img_channel=3, width=TEACHER_MODEL[0]['width'], middle_blk_num=TEACHER_MODEL[0]['middle_blk_num'],enc_blk_nums=TEACHER_MODEL[0]['enc_blk_nums'], dec_blk_nums=TEACHER_MODEL[0]['dec_blk_nums'])
    print('model:', teacher_model)
    summary(teacher_model, (3, 512, 512), batch_size=4, device='cpu')
    # weights = '../work_dirs/shuangxiaba/0616/skin_distill_gradient_68to1M/models/model_iter_best.pth'

    # weights = torch.load(weights)['state_dict']
    # weights_dict = {}
    # for k, v in weights.items():
    #     new_k = k.replace('module.', '') if 'module' in k else k
    #     weights_dict[new_k] = v
    # teacher_model.load_state_dict(weights_dict)
        



    print('model.ending:{}\n model.decoder:{}'.format(teacher_model.ending, teacher_model.decoders))


    # x = (4, 3, 512, 512)
    # feature = teacher_model.ending(x)
    
    # print('teacher_model.ending:{}, feature:{}'.format(teacher_model.ending, feature))

    # for name, module in teacher_model.named_modules():
    #     print('name:', name)


    # import torchextractor as tx
    # tea_model = tx.Extractor(teacher_model, ['decoders.3.0', 'ending'])
    # x = torch.rand(4, 3, 512, 512)
    # model_output, features = tea_model(x)
    # feature_shapes = {name:f.shape for name, f in features.items()}
    # print('feature_shapes:', feature_shapes)
