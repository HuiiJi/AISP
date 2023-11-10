import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile


class Upsample2D(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, scale_size:int = None):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_size = scale_size
        up = [
            nn.Upsample(scale_factor=2, mode='nearest') if scale_size is None else nn.Upsample(size=scale_size, mode='nearest'),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        ]
        self.up = nn.ModuleList(up)
        
        
    def forward(self, x):
        for layer in self.up:
            x = layer(x)
        return x


class Downsample2D(nn.Module):
    """
    this is a downsample block with one conv2d layer and conv2d stride = 2, the input and output channels are the same
    
    Args:
        in_channels: the input channels of the first conv2d layer
    """
    def __init__(self, in_channels:int):
        super().__init__()
        down = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
        ]
        self.down = nn.ModuleList(down)
        
        
    def forward(self, x):
        for layer in self.down:
            x = layer(x)
        return x
    

    
class Unet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(Unet, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
        )
        self.pool1 = Downsample2D(32)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
        )
        self.pool2 = Downsample2D(64)

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
        )
        self.pool3 = Downsample2D(128)

        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
        )
        self.pool4 = Downsample2D(256)

        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
        )

        self.upv6 = Upsample2D(512, 256)
        self.conv_6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
        )

        self.upv7 = Upsample2D(256, 128)
        self.conv_7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
        )
        self.upv8 = Upsample2D(128, 64)
        self.conv_8 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
        )

        self.upv9 = Upsample2D(64, 32)
        self.conv_9 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
        )

        self.conv_10 = nn.Conv2d(32, out_channels, kernel_size=(1,1), stride=(1,1))


    def forward_features(self, inputs: torch.Tensor):
        n, c, h, w = inputs.shape
        h_pad = 32 - h % 32 
        w_pad = 32 - w % 32 
        inputs = F.pad(inputs, (0, w_pad, 0, h_pad), 'constant')
        return inputs
    

    def forward(self, inputs: torch.Tensor):
        n, c, h, w = inputs.shape
        inp = self.forward_features(inputs)
        # down
        conv1 = self.conv_1(inp)
        pool1 = self.pool1(conv1)

        conv2 = self.conv_2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv_3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv_4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv_5(pool4)

        # up
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv_6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv_7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv_8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv_9(up9)

        conv10 = self.conv_10(conv9)
        out = inp + conv10
        out = out[..., :h, :w]
        return out


if __name__ == "__main__":
    flops, params = profile(Unet(), inputs = (torch.randn(1, 4, 640, 960), ))
    print('MACs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M' )
pass

