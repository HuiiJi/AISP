import torch
import torch.nn as nn
import os
import torch.nn.functional as F


################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=(1,1)):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


############################# SAM ##############################
# 通道注意层
# 组成CAB
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 特征进行先下采样然后上采样
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, (1,1), padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, (1,1), padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# 通道注意块
# conv+CAB----> shallow_feat
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


# 监督注意模块
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 4, kernel_size, bias=bias)  # 原本为3
        self.conv3 = conv(4, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


################################ Down/Up Sample ################################
# 进行上下采样
# scale_factor:缩放因子，>1 上采样，<1下采样
# model：上采样算法，使用bilinear双线性插值
# align_corners – 如果为True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值。
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, (1,1), stride=(1,1), padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


# 使用双插值这有助于减少由于转置卷积而经常出现的输出图像中的棋盘状伪影
class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, (1,1), stride=(1,1), padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


# 用在U-Net子网络中
class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, (1,1), stride=(1,1), padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


######################################### U-Net #########################################
# 编码器
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()
        # CAB为了提取特征
        # 译码器阶段一：两个CAB
        # 输入通道：n_feat
        self.encoder_level1 = [CAB(n_feat,                         kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        # 阶段二：两个CAB
        # 输入通道改为n_feat + scale_unetfeats，原因：下采样后增加通道数。U-Net论文有解释
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats,       kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        # 阶段二：两个CAB
        # 输入通道改为n_feat + scale_unetfeats*2
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat,                   scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # 跨阶段融合(CSFF)
        # 个人理解：跨阶段融合其实是将前一阶段的feat和res与本阶段的cat一起输入encoder中，详情可以看流程图
        # feat是译码器输出   res是解码器输出
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                         n_feat,                         kernel_size=(1,1), bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats,       n_feat + scale_unetfeats,       kernel_size=(1,1), bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=(1,1), bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                         n_feat,                         kernel_size=(1,1), bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats,       n_feat + scale_unetfeats,       kernel_size=(1,1), bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=(1,1), bias=bias)

    # encoder_out：前一阶段译码器输出
    # decoder_out：前一阶段解码器输出
    # enc1：进入下一阶段即SAM12
    # 在跨阶段融合中使用enc1、2、3
    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


# 解码器
class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                         kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats,       kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                   kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat,                   scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


######################################### ORSNet #########################################
# 原始分辨率解析块(ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# 原始分辨率子网络
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat,                   scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat,                   scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=(1,1), bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=(1,1), bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=(1,1), bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=(1,1), bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=(1,1), bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=(1,1), bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


######################################### MPRNet #########################################
# 跨阶段特征融合：前一阶段的encode和decode相加的结果和本阶段编码器的输出相加
# 跨阶段特征融合优点：1、减少由于多次上下采样造成的信息损失。2、一个阶段的多尺度信息有助下一阶段的特征。3、减轻数据流动，导致模型优化更稳定。
class MPRNet(nn.Module):
    # n_feat=96, scale_unetfeats=48, scale_orsnetfeats=32
    def __init__(self, in_c=4, out_c=4, n_feat=64, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super(MPRNet, self).__init__()

        act = nn.PReLU()

        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        # 初始阶段不需要跨阶段特征融合
        # 返回有三个参数
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        # 与第一个阶段的代码进行特征融合
        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        # 阶段三
        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)

        self.concat12 = conv(n_feat * 2, n_feat,                     kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, x3_img):
        n, c, h, w = x3_img.shape
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        x3_img = F.pad(x3_img, (0, w_pad, 0, h_pad), 'replicate')

        # 获取图像尺寸用来分割
        H = x3_img.size(2)
        W = x3_img.size(3)
        # 分割图片
        # Two Patches for Stage 2  (stage2 的两个切片)
        x2top_img = x3_img[:, :, 0:int(H / 2), :]
        x2bot_img = x3_img[:, :, int(H / 2):H, :]
        # Four Patches for Stage 1 (stage1 的四个切片)
        x1ltop_img = x2top_img[:, :, :, 0:int(W / 2)]  # 左上角
        x1rtop_img = x2top_img[:, :, :, int(W / 2):W]  # 右上角
        x1lbot_img = x2bot_img[:, :, :, 0:int(W / 2)]  # 左下角
        x1rbot_img = x2bot_img[:, :, :, int(W / 2):W]  # 右下角
        ##-------------- Stage 1---------------------
        # 计算表面特征
        x1ltop = self.shallow_feat1(x1ltop_img)  # torch.Size([1, 96, 256, 256])
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)
        # 四块图像进行译码操作
        feat1_ltop = self.stage1_encoder(x1ltop)  # torch.Size([1, 96, 256, 256])
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)
        # 拼合深层特征，拼合后成为上下两块 # torch.Size([1, 96, 256, 512])
        feat1_top = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]
        # Decoder
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)
        ## 使用SAM
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)
        # 获取阶段一的图像
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)
        ##-------------- Stage 2---------------------
        ## 计算表层特征
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)
        # 连接阶段1的SAM特征和阶段2的表层特征，通道融合
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)
        ## 连接深层特征
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]
        res2 = self.stage2_decoder(feat2)
        ## 使用SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)
        ##-------------- Stage 3---------------------
        ## 计算表层特征
        x3 = self.shallow_feat3(x3_img)
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)
        stage3_img = self.tail(x3_cat)

        output1 = stage1_img[:, :, :h, :w]
        output2 = stage2_img[:, :, :h, :w]
        output3 = stage3_img + x3_img
        output3 = output3[:, :, :h, :w]

        return [output3, output2, output1]
 
if __name__ == '__main__':
    from thop import profile
    flops, params = profile(MPRNet(), inputs = (torch.randn(1, 4, 540, 960), ))
    print('MACs = ' + str(flops/1000**3) + 'G' + '(MPR)')
    print('Params = ' + str(params/1000**2) + 'M' + '(MPR)')
pass
