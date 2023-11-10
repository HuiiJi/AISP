#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/10/15
# @Description: generate noisy_data class


from noise_profile import Noise_Calibration
import numpy as np
import lmdb
import glob
from typing import Any, Dict, List, Optional, Tuple, Union
import os
from pathlib import PosixPath, Path
from tqdm import tqdm
from tools import FeedRAW, normalization, inv_normalization, Bayer_unify, Bayer2RGGB, RGGB2Bayer


class Generate_train_lmdb(object):
    """
    Generate lmdb dataset for denoise training
    
    description:
        generate lmdb dataset for denoise training
    
    usage:
        Generat_train_lmdb(input_path, lmdb_path, stride, noise_calibration)
    """
    def __init__(self, 
        noise_calibration: Noise_Calibration,
        stride: int = 128):
        super().__init__()
        """
        input_path: str, input path of raw images
        lmdb_path: str, output path of lmdb dataset
        stride: int, stride of sliding window
        noise_calibration: Noise_Calibrationnoise calibration class
        """
        self.stride = stride
        self.noise_params = noise_calibration.run()
        self.root_path = noise_calibration.root_path
        self.noise_model = noise_calibration.noise_model
        self.lmdb_path = self.root_path /  'IMX766' / 'train_data'
        self.gt_path = self.root_path /  'IMX766' / 'train_data' / 'gt_img'
        self.train_data_idx = self.root_path / 'IMX766' / 'train_data' / 'train_data_idx.txt'
        self.blc = 1024
        self.wlc = 16383
        self.__check_inputs()
  
     
    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        set attribute
        """
        self.__dict__[__name] = __value
    
    
    def __check_inputs(self) -> None:
        """
        check inputs
        """
        if not isinstance(self.stride, int):
            raise TypeError(f'stride should be int, please check it, now is {type(self.stride)}')
        if not os.path.exists(self.gt_path):
            raise TypeError(f'input_path not exists, please check it, now is {self.gt_path}')
        if not os.path.isdir(self.lmdb_path):
            raise TypeError(f'lmdb_path should be a dir, please check it, now is {self.lmdb_path}')
        if not isinstance(self.gt_path, str):
            self.input_path = str(self.gt_path)
        assert self.blc is not None, 'blc is None, please check it'
        assert self.wlc is not None, 'wlc is None, please check it'
        assert self.noise_params is not None, 'noise_params is None, please check it'
        assert self.stride is not None, 'stride is None, please check it'  
        os.makedirs(self.lmdb_path, exist_ok=True)
        self.env = lmdb.open(str(self.lmdb_path), map_size=int(1e10))

    
    def run(self)-> None:
        """
        Generate noisy images from clean images
        """
        __noise_model_dict__ = {
            'pg_model': self.__generate_pg_noisy(input_path = self.input_path, env = self.env, stride = self.stride),
            'pg_tl_model': self.__generate_pg_tl_noisy(input_path = self.input_path, env = self.env, stride = self.stride),
            
        }
        return __noise_model_dict__.pop(self.noise_model)
        
        
    def __generate_pg_noisy(self, input_path:str, env: lmdb.Environment, stride: int) -> None:
        """
        Generate noisy images from clean images
        
        Args:
            inputs_path: str, path to clean images
            env: lmdb.Environment, lmdb environment
            stride: int, stride of patch
        """
        clean_img_list = glob.glob(input_path + '/*')
        feedraw = [FeedRAW(img).raw_data for img in clean_img_list]
        raw_unify = [Bayer_unify(raw=img, input_pattern='BGGR', target_pattern='RGGB', mode='pad') for img in feedraw]
        with open(str(self.train_data_idx), 'w') as f:
            for idx, clean_img in tqdm(enumerate(raw_unify)):
                H, W = clean_img.shape[:2]
                for h0 in range(0, int(H / stride)):
                    for w0 in range(0, int(W / stride)):
                        split_clean_bayer = clean_img[h0 * stride:h0 * stride + stride, w0 * stride:w0 * stride + stride]
                        split_clean_rggb = Bayer2RGGB(split_clean_bayer)
                        split_clean_norm = normalization(split_clean_rggb, black_level = self.blc, white_level = self.wlc)
                        """
                        Add noise and convert to Bayer
                        """
                        iso_selected = np.random.randint(100, 6400)
                        shot_noise = np.poly1d(list(self.noise_params[0]))(iso_selected)
                        read_noise = np.poly1d(list(self.noise_params[1]))(iso_selected)
                        split_noisy_norm = self.__add_pg_noise(split_clean_norm, shot_noise = shot_noise, read_noise = abs(read_noise))
                        """
                        Save noisy images to lmdb
                        """
                        self.__insert(env, f'gt_idx_{idx}_iso_{iso_selected}', split_clean_norm.tobytes())
                        self.__insert(env, f'noisy_idx_{idx}_iso_{iso_selected}', split_noisy_norm.tobytes())
                        f.write(f'idx_{idx}_iso_{iso_selected}' + '\n')
        f.close()
        env.close()    
        print(f' - successfully - train data is saved in {self.lmdb_path}, and the index is saved in {self.train_data_idx}')
        
    
    def __generate_pg_tl_noisy(self, input_path:str, env: lmdb.Environment, stride: int) -> None:
         """
         will be updated later
         """
         pass
         
        
    def __insert(self, env: lmdb.Environment, sid: str, name: bytes) -> None:
        """
        insert the image into lmdb database
        
        Args:
            env:
                lmdb environment
            sid:
                image id
            name:
                image name
        """
        txn = env.begin(write=True)
        txn.put(str(sid).encode('ascii'), name)
        txn.commit()


    def __add_pg_noise(self, inputs:np.ndarray, shot_noise:float, read_noise:float) -> np.ndarray:
        """
        add poisson-gaussian noise to the raw image
        
        Args:
            inputs:
                input raw image, in shape (H, W)
            shot_noise:
                shot noise, default 0.01
            read_noise:
                read noise, default 0.01
        """
        assert inputs.dtype == np.float32, 'input should be float32!'
        assert shot_noise >= 0 and read_noise >= 0, 'shot noise and read noise should be non-negative!'
        noisy_img = np.random.poisson(inputs / shot_noise) * shot_noise + np.random.normal(0, read_noise, inputs.shape)
        return np.clip(noisy_img, 0, 1).astype(np.float32)
    

if __name__ == "__main__":
    Generate_train_lmdb(noise_calibration=Noise_Calibration()).run()
    

