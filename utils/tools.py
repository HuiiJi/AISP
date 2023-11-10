#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/10/10
# @Description: tools


from typing import Any, Dict, List, Optional, Tuple, Union
import rawpy
import os
import numpy as np
from pathlib import PosixPath, Path
import torch
import torch.distributed as dist
import yaml
import time


class FeedRAW(object):
    """
    Feed In RAW Image
    
    description:
        this is a class for feed in the raw image
    
    step:
        1. get the raw image
        2. preprocess the raw image
        
    usage:
        raw = FeedRAW(raw_img_path, Height=4032, Width=3024)
    """
    def __init__(self, raw_img_path:str= None, raw_height: int = None, raw_width: int = None) -> None:
        super().__init__()
        self.raw_img_path = raw_img_path
        self.raw_height = raw_height
        self.raw_width = raw_width
        self.__check_inputs()
        self.raw_img_dtype = 'Metadata' if self.raw_img_path.split('.')[-1] in ('dng', 'DNG','nef', 'NEF', 'cr2', 'CR2') else 'NoMetadata'
        self.meta_data = self.__get_metadata() if self.raw_img_dtype == 'Metadata' else None
        self.raw_data = self.run()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.raw_img_path is not None, 'raw_img_path is None, please check it'
        if isinstance(self.raw_img_path, PosixPath):
            self.raw_img_path = str(self.raw_img_path)
        if not isinstance(self.raw_img_path, str):
            raise TypeError(f'raw_img_path should be str, please check it, now is {type(self.raw_img_path)}')
        if self.raw_img_path.split('.')[-1] not in ('dng', 'DNG', 'raw', 'RAW', 'nef', 'NEF', 'cr2', 'CR2', 'tif', 'TIFF'):
            raise TypeError(f'RAW image should be dng, DNG, raw, RAW, nef, NEF, cr2, CR2, tif, TIFF, please check it, now is {self.raw_img_path.split(".")[-1]}')
        if not os.path.exists(self.raw_img_path):
            raise TypeError(f'RAW image path not exists, please check it, now is {self.raw_img_path}')
        
    
    def run(self) -> np.ndarray:
        """
        get the raw image
        """
        __dict__ = {
            'Metadata': self.__get_raw_with_metadata,
            'NoMetadata': self.__get_raw_without_metadata
        }
        return __dict__.pop(self.raw_img_dtype)()
    
    
    def __get_raw_with_metadata(self) -> np.ndarray:
        """
        get the raw image with metadata, such as .dng, .DNG, .nef, .NEF, .cr2, .CR2
        """
        raw = rawpy.imread(self.raw_img_path)
        raw_img = raw.raw_image_visible.astype(np.uint16)
        del raw
        return raw_img
    
    
    def __get_metadata(self)-> Dict[str, Any]:  
        """
        get the metadata of the raw image, such as .dng, .DNG, .nef, .NEF, .cr2, .CR2
        """
        assert self.raw_img_dtype == 'Metadata', 'raw_img_dtype should be Metadata, please check it'
        raw = rawpy.imread(self.raw_img_path)
        metadata = dict()
        metadata['black_level'] = raw.black_level_per_channel[0]
        metadata['white_level'] = raw.white_level
        metadata['bayer_pattern'] = self.__get_bayer_pattern(raw)
        return metadata
    
    
    def __get_bayer_pattern(self, raw:rawpy.RawPy) -> str:
        """
        get the bayer pattern
        
        Args:
            raw:   rawpy.RawPy
        """
        bayer_desc = 'RGBG'
        bayer_pattern = ''
        bayer_pattern_matrix = raw.raw_pattern
        if bayer_pattern_matrix is not None:
            for i in range(0, 2):
                for k in range(0, 2):
                    bayer_pattern += (bayer_desc[bayer_pattern_matrix[i][k]])
            else:
                bayer_pattern = 'RGGB'
        return bayer_pattern


    def __get_raw_without_metadata(self) -> np.ndarray: 
        """
        get the raw image without metadata, such as .raw, .RAW
        """
        assert self.raw_height is not None, 'raw_height is None, please check it'
        assert self.raw_width is not None, 'raw_width is None, please check it'
        assert self.raw_img_dtype == 'NoMetadata' 
        return np.fromfile(self.raw_img_path, dtype=np.uint16).reshape((self.raw_height, self.raw_width))
    
    
def normalization(inputs:np.ndarray, black_level:float, white_level:float) -> np.ndarray:
    """
    normlization the raw image
    
    Args:
        inputs:
            input raw image, in shape (H, W)
        black_level:
            black level, default 0
        white_level:
            white level, default 1023
    """
    return np.maximum(inputs.astype(np.float32) - black_level, 0) / (white_level - black_level)
    
    
def inv_normalization(inputs:np.ndarray, black_level:float, white_level:float) -> np.ndarray:
    """
    inverse normlization the raw image
    
    Args:
        inputs:
            input raw image, in shape (H, W)
        black_level:
            black level, default 0
        white_level:
            white level, default 1023
    """
    return np.minimum(inputs * (white_level - black_level) + black_level, white_level).astype(np.uint16)


def RGGB2Bayer(input_data: np.ndarray) -> np.ndarray:
    """
    RGGB to bayer image
    
    Args:
        input_data: np.ndarray in shape (H, W, 4)
    """
    assert input_data.shape[-1] == 4, 'input_data should be in shape (H, W, 4), please check it'
    height, width = input_data.shape[:2]
    raw_data = np.zeros((height*2, width*2), dtype = np.uint16)
    raw_data[0: height*2: 2, 0: width*2: 2] = input_data[:, :, 0]
    raw_data[0: height*2: 2, 1: width*2: 2] = input_data[:, :, 1]
    raw_data[1: height*2: 2, 0: width*2: 2] = input_data[:, :, 2]
    raw_data[1: height*2: 2, 1: width*2: 2] = input_data[:, :, 3]
    return raw_data
    

def Bayer2RGGB(input_data: np.ndarray) -> np.ndarray:
    """
    bayer image to RGGB
    
    Args:
        input_data: np.ndarray in shape (H, W)
            bayer image
    """
    assert len(input_data.shape) == 2, 'input_data should be 2D array, please check it'
    height, width = input_data.shape[:2]
    raw_data = np.zeros((height//2, width//2, 4), dtype = np.uint16)
    raw_data[:, :, 0] = input_data[0: height: 2, 0: width: 2]
    raw_data[:, :, 1] = input_data[0: height: 2, 1: width: 2]
    raw_data[:, :, 2] = input_data[1: height: 2, 0: width: 2]
    raw_data[:, :, 3] = input_data[1: height: 2, 1: width: 2]
    return raw_data
         
    
def Bayer_unify(raw: np.ndarray, input_pattern: str, target_pattern: str, mode: str) -> np.ndarray:
    """
    Convert a bayer raw image from one bayer pattern to another.

    Args:
        raw : np.ndarray in shape (H, W)
            Bayer raw image to be unified.
        input_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
            The bayer pattern of the input image.
        target_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
            The expected output pattern.
        mode: {"crop", "pad"}
            The way to handle submosaic shift. "crop" abandons the outmost pixels,
            and "pad" introduces extra pixels. Use "crop" in training and "pad" in
            testing.
    """
    BAYER_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]
    NORMALIZATION_MODE = ["crop", "pad"]
    if input_pattern not in BAYER_PATTERNS:
        raise ValueError('Unknown input bayer pattern!')
    if target_pattern not in BAYER_PATTERNS:
        raise ValueError('Unknown target bayer pattern!')
    if mode not in NORMALIZATION_MODE:
        raise ValueError('Unknown normalization mode!')
    if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
        raise ValueError('raw should be a 2-dimensional numpy.ndarray!')

    if input_pattern == target_pattern:
        h_offset, w_offset = 0, 0
    elif input_pattern[0] == target_pattern[2] and input_pattern[1] == target_pattern[3]:
        h_offset, w_offset = 1, 0
    elif input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]:
        h_offset, w_offset = 0, 1
    elif input_pattern[0] == target_pattern[3] and input_pattern[1] == target_pattern[2]:
        h_offset, w_offset = 1, 1
    else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
        raise RuntimeError('Unexpected pair of input and target bayer pattern!')

    if mode == "pad":
        out = np.pad(raw, [[h_offset, h_offset], [w_offset, w_offset]], 'reflect')
    elif mode == "crop":
        h, w = raw.shape
        out = raw[h_offset:h - h_offset, w_offset:w - w_offset]
    else:
        raise ValueError('Unknown normalization mode!')
    return out
    
    
def simple_isp(input_data : np.ndarray, gamma = 0.45)->np.ndarray:
    assert len(input_data.shape) == 4 or input_data.shape[2] == 4, 'Input data shape does not match metadata, input data should be 2D or 3D with 1 channel, but got {}'.format(input_data.shape)
    raw_data = input_data[0].cpu().numpy() if input_data.dtype == torch.float32 else input_data[0]
    rgb_data = np.stack((raw_data[0, ...], np.mean(raw_data[1:3, ...], 0), raw_data[3, ...]), 0)
    rgb_norm = (rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min())
    rgb_norm[0, ...] = rgb_norm[0, ...] * 1.6
    rgb_norm[2, ...] = rgb_norm[2, ...] * 1.3
    rgb_gma = rgb_norm ** (gamma)
    rgb = np.clip(rgb_gma * 255, 0, 255)
    return rgb.transpose(1, 2, 0).astype(np.uint8)

    
def singleton(cls, *args, **kwargs) -> object:
    """
    singleton pattern
    """
    _instance = {}
    def _singleton():
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return _singleton


def set_random_seed(seed:int) -> None:
    """
    set random seed for numpy and torch
    
    Args:
        seed:random seed
    """
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   


def reduce_mean(tensor: torch.Tensor, nprocs: int) -> torch.Tensor:
    """
    reduce mean
    
    Args:
        tensor: tensor to be reduced
        nprocs: number of processes
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt
    
    
def set_config(config:str) -> None:
    """
    set args
    """
    def from_yaml(yaml_path:str) -> Dict:
        """ 
        Instantiation from a yaml file. 
        """
        if not isinstance(yaml_path, str):
            yaml_path = str(yaml_path)
        with open(yaml_path, 'r') as fp:
            yml = yaml.safe_load(fp)
        return yml
    return from_yaml(config)

    
def time_cost_decorator(func):
    """
    Decorator for time cost, print the time cost of the function
    """
    def warp(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        end_time = time.time()
        print('|',f'{func.__name__}'.ljust(50), f': cost time is {1000 * (end_time - start_time):.2f} ms'.ljust(20),'|') 
        print('-'* 88)
        return out
    return warp


def save_valid_image(input_path:str, image:np.ndarray):
    """
    """
    import cv2
    cv2.imwrite(input_path, image[..., ::-1])
    

def calculate_psnr(img1: torch.tensor, img2: torch.tensor, dmax:float = 1.0) -> float:
    """
    calculate psnr
    
    Args:
        img1: image 1
        img2: image 2
    """
    assert img1.shape == img2.shape, 'Image shape does not match'
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(dmax ** 2 / mse)


def calculate_ssim(img1: torch.tensor, img2: torch.tensor, dmax:float = 1.0) -> float:
    """
    calculate ssim
    
    Args:
        img1: image 1
        img2: image 2
    """
    assert img1.shape == img2.shape, 'Image shape does not match'
    c1 = (0.01 * dmax) ** 2
    c2 = (0.03 * dmax) ** 2
    mean_x = torch.mean(img1)
    mean_y = torch.mean(img2)
    std_x = torch.std(img1)
    std_y = torch.std(img2)
    cov = torch.mean((img1 - mean_x) * (img2 - mean_y))
    ssim = (2 * mean_x * mean_y + c1) * (2 * cov + c2) / ((mean_x ** 2 + mean_y ** 2 + c1) * (std_x ** 2 + std_y ** 2 + c2))
    return ssim


class Charbon_loss(torch.nn.Module):
    """L1 Charbonnierloss"""
    def __init__(self, eps:float = 1e-3) -> None:
        super(Charbon_loss, self).__init__()
        self.eps = eps
        
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff ** 2 + (self.eps ** 2)))
        

        
      
