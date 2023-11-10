import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math


class Upsample2D(nn.Module):
    """
    this is a upsample block with one conv2d layer and nearest upsample, the input and output channels are the same
    
    Args:
        in_channels: the input channels of the first conv2d layer
        scale_size: the scale size of the upsample layer
    """
    def __init__(self, in_channels:int, scale_size:Optional[int] = None):
        super().__init__()
        self.scale_size = scale_size
        up = [
            nn.Upsample(scale_factor=2, mode='nearest') if scale_size is None else nn.Upsample(size=scale_size, mode='nearest'),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
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
        

class ResnetBlock2D(nn.Module):
    """
    this is a resnet block with two conv2d layers, the input and output channels are the same
    
    Args:
        in_channels: the input channels of the first conv2d layer
        out_channels: the output channels of the second conv2d layer
        kernel_size: the kernel size of the conv2d layers
        stride: the stride of the conv2d layers
    """
    def __init__(self, in_channels:int, out_channels:Optional[int] = None, kernel_size:int = 3, stride:int = 1, padding:int = 1):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        resnet_block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),  
        ]
        self.resnet_block = nn.ModuleList(resnet_block)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()
        
        
    def forward(self, x:torch.tensor):
        residual = x
        for layer in self.resnet_block:
            x = layer(x)
        x = x + self.conv_shortcut(residual) 
        return x


class SelfAttnBlock2D(nn.Module):
    """
    Self attention block for 2D image
    
    Args:
        in_channels: input channels
        num_head_channel: number of channels for each head
    """
    def __init__(self, in_channels:int, num_head_channel:Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = in_channels // num_head_channel if num_head_channel is not None else 1
        self.group_norm = nn.GroupNorm(num_channels=in_channels, num_groups=32, eps=1e-5, affine=True)
        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_kv = nn.Linear(in_channels, in_channels * 2)
        self.proj =nn.Linear(in_channels, in_channels)
        
        
    def __reshape_heads_to_batch(self, hidden_states:torch.tensor):
        b, seq, channel = hidden_states.shape
        hidden_states = hidden_states.reshape(b, seq, self.num_heads, channel // self.num_heads)
        tensor = hidden_states.permute(0, 2, 1, 3).reshape(b * self.num_heads, seq, channel // self.num_heads)
        return tensor
    
    
    def __reshape_batch_to_heads(self, hidden_states:torch.tensor):
        b, seq, channel = hidden_states.shape
        hidden_states = hidden_states.reshape(b // self.num_heads, seq, self.num_heads, channel)
        tensor = hidden_states.permute(0, 2, 1, 3).reshape(b // self.num_heads, seq, channel * self.num_heads)
        return tensor
    
    
    def forward(self, x:torch.tensor):
        resudial = x
        b, c, h, w = x.shape
        hidden_states = self.group_norm(x)
        hidden_states = hidden_states.view(b, c, h * w).transpose(-1, -2)
        query = self.to_q(hidden_states)
        key, value = self.to_kv(hidden_states).chunk(2, dim=2)
        query = self.__reshape_heads_to_batch(query)
        key = self.__reshape_heads_to_batch(key)
        value = self.__reshape_heads_to_batch(value)
        scale = 1 / math.sqrt(self.in_channels / self.num_heads)
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0.0,
            alpha=scale
        )
        # attention_scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) 
        attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)
        hidden_states = torch.bmm(attention_probs, value)
        # hidden_states = torch.matmul(attention_probs, value)
        hidden_states = self.__reshape_batch_to_heads(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        hidden_states = hidden_states + resudial
        return hidden_states
        

class DownBlock2D(nn.Module):
    """
    this block is used for downsample hidden states, and output hidden states for skip connection
    
    Args:
        in_channels: input channels
        out_channels: output channels
        num_layer: number of resnet block in this block
    """
    def __init__(self, in_channels:int, out_channels:int, num_layer:int = 2):
        super().__init__()
        resnet = []
        for i in range(0, num_layer):
            in_channels = in_channels if i == 0 else out_channels
            resnet.append(
                ResnetBlock2D(in_channels, out_channels)
            )   
        self.resnet = nn.ModuleList(resnet)
        self.downsample = Downsample2D(out_channels)
     
        
    def forward(self, hidden_states:torch.tensor):
        output_states = ()
        for resnet in self.resnet:
            hidden_states = resnet(hidden_states)
            output_states += (hidden_states,)
        hidden_states = self.downsample(hidden_states)
        output_states += (hidden_states,)
        return hidden_states, output_states
    
        
class DownSelfAttnBlock2D(nn.Module):
    """
    this block is used in the encoder part of the transformer, it contains a self attention block and a downsample block

    Args:
        in_channels: input channels
        num_layer: number of resnet block in this block
        num_head_channel: number of head channel in the self attention block
    """
    def __init__(self, in_channels:int, num_layer:int = 2, num_head_channel = 32):
        super().__init__()
        resnet = []
        self_attn = []
        for i in range(0, num_layer):
            resnet.append(
                ResnetBlock2D(in_channels)
            )
            self_attn.append(
                SelfAttnBlock2D(in_channels, num_head_channel)
            )
        self.resnet = nn.ModuleList(resnet)
        self.self_attn = nn.ModuleList(self_attn)
      
               
    def forward(self, hidden_states:torch.tensor):
        output_states = ()
        for resnet, self_attn in zip(self.resnet, self.self_attn):
            hidden_states = resnet(hidden_states)
            hidden_states = self_attn(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class MidBlock2D(nn.Module):
    """
    this block is used for mid layers, so the input and output channels are the same
    
    Args:
        in_channels: input channels
        out_channels: output channels
        num_layer: number of resnet blocks
    """
    def __init__(self, in_channels:int, num_layer:int = 2):
        super().__init__()
        resnet = []
        for i in range(0, num_layer):
            resnet.append(
                ResnetBlock2D(in_channels)
            )   
        self.resnet = nn.ModuleList(resnet)
     
        
    def forward(self, hidden_states:torch.tensor):
        for resnet in self.resnet:
            hidden_states = resnet(hidden_states)
        return hidden_states
    

class UpSelfAttnBlock2D(nn.Module):
    """
    this block is used for upsample, so the input and output channels are the same
    
    Args:
        in_channels: input channels
        out_channels: output channels
        num_layer: number of resnet blocks
        use_upsample: whether to use upsample
    """
    def __init__(self, in_channels:int, out_channels:int, num_layer:int = 3, use_upsample:bool = True):
        super().__init__()
        resnet = []
        self_attn = []
        for i in range(0, num_layer):
            resnet_in_channels = out_channels + in_channels 
            resnet.append(
                ResnetBlock2D(resnet_in_channels, out_channels)
            )
            self_attn.append(
                SelfAttnBlock2D(out_channels)
            )
        self.resnet = nn.ModuleList(resnet)
        self.self_attn = nn.ModuleList(self_attn)
        self.upsample = Upsample2D(in_channels) if use_upsample else nn.Identity()
        
    
    def forward(self, hidden_states:torch.tensor, res_tuple:Tuple):
        for resnet, self_attn in zip(self.resnet, self.self_attn):
            res_states = res_tuple[-1]
            res_tuple = res_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_states], dim=1)
            hidden_states = resnet(hidden_states)
            hidden_states = self_attn(hidden_states)
        hidden_states = self.upsample(hidden_states)
        return hidden_states
    
    
class UpBlock2D(nn.Module):
    """ 
    this block is used for upsample, so the input and output channels are the same
    """
    def __init__(self, in_channels:int, skip_in_channels:int, out_channels:int, num_layer:int = 3, use_upsample:bool = True):
        super().__init__()
        resnet = []
        for i in range(0, num_layer):
            pre_in_channels = skip_in_channels if i == num_layer - 1 else out_channels
            pre_out_channels = in_channels if i == 0 else out_channels
            resnet_in_channels = pre_in_channels + pre_out_channels
            resnet.append(
                ResnetBlock2D(resnet_in_channels, out_channels)
            )
        self.resnet = nn.ModuleList(resnet)
        self.upsample = Upsample2D(out_channels) if use_upsample else nn.Identity()
        
    
    def forward(self, hidden_states:torch.tensor, res_tuple:Tuple):
        for resnet in self.resnet:
            res_states = res_tuple[-1]
            res_tuple = res_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_states], dim=1)
            hidden_states = resnet(hidden_states)
        hidden_states = self.upsample(hidden_states)
        return hidden_states
            

class My_network(nn.Module):
    """
    this is the main network, it contains the encoder and decoder part, and the final conv layer, the input and output channels are the same
    
    Args:
        in_channels: input channels
        out_channels: output channels
        conv_in_channels: input channels of the first conv layer
        block_out_channels: output channels of each block
    """
    def __init__(self, 
        in_channels:int = 4, 
        out_channels:int = 4,
        conv_in_channels:int = 8,
        block_out_channels:Tuple = (8, 16, 32)):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, conv_in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, stride=1, padding=1)
        down_block = []
        up_block = []
        mid_block = []
        oc = conv_in_channels
        for i in range(0, len(block_out_channels)):
            ic = oc
            oc = block_out_channels[i]
            down_block.append(
                DownBlock2D(ic, oc)
            )
        down_block.append(DownSelfAttnBlock2D(oc))
        mid_block.append(MidBlock2D(block_out_channels[-1]))
        reverse_block_out_channels = block_out_channels[::-1]
        oc = reverse_block_out_channels[0]
        up_block.append(UpSelfAttnBlock2D(oc, oc))
        for i in range(0, len(reverse_block_out_channels)):
            ic = oc 
            skip_c = reverse_block_out_channels[i + 1] if i != len(reverse_block_out_channels) - 1 else reverse_block_out_channels[i]
            oc = reverse_block_out_channels[i]
            up_block.append(
                UpBlock2D(ic, skip_c, oc) if i != len(reverse_block_out_channels) - 1 else UpBlock2D(ic, skip_c, oc, use_upsample=False)
            )
        self.down_block = nn.ModuleList(down_block)
        self.mid_block = nn.ModuleList(mid_block)
        self.up_block = nn.ModuleList(up_block)
        
        
    def forward_features(self, inputs: torch.Tensor):
        n, c, h, w = inputs.shape
        h_pad = 32 - h % 32 if h % 32 != 0 else 0
        w_pad = 32 - w % 32 if w % 32 != 0 else 0
        inputs = F.pad(inputs, (0, w_pad, 0, h_pad), 'constant')
        return inputs
    
    
    def forward(self, inputs:torch.tensor):
        n, c, h, w = inputs.shape
        inp = self.forward_features(inputs)
        hidden_states = self.conv_in(inp)
        out_res_tuple = (hidden_states, )
        for down_block in self.down_block:
            hidden_states, res = down_block(hidden_states)
            out_res_tuple += res
        for mid_block in self.mid_block:
            hidden_states = mid_block(hidden_states)
        for up_block in self.up_block:
            hidden_states = up_block(hidden_states, out_res_tuple)
            out_res_tuple = out_res_tuple[:-3] 
        hidden_states = self.conv_out(hidden_states)
        hidden_states = inp + hidden_states
        hidden_states = hidden_states[..., :h, :w]
        return hidden_states

            
    
if __name__ == "__main__":
    device = torch.device('cuda')
    inputs = torch.randn(1, 4, 256, 256).to(device)
    network = My_network().to(device)
    out = network(inputs)
    print(out.shape)
 