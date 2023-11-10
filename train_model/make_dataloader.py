#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/10/20
# @Description: dataset class


import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import lmdb
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tools import FeedRAW, normalization, inv_normalization, Bayer_unify, Bayer2RGGB, RGGB2Bayer
from typing import Any, Dict, List, Optional, Tuple, Union


class Trainset(Dataset):
    """
    Trainset, read from lmdb and return noisy and clean image
    """
    def __init__(self) -> None:
        """
        root_path:
            root path of the project
        train_lmdb_path:
            path of the lmdb file
        train_idx_path:
            path of the index file
        id_list:
            list of the index
        """
        self.root_path = Path(os.path.abspath(__file__)).parent.parent
        self.train_lmdb_path = self.root_path  / 'IMX766' / 'train_data'
        self.id_list = open(self.root_path  /  'IMX766' / 'train_data' / 'train_data_idx.txt', 'r').read().split()
        self.train_id = self.id_list[:-5]
        self.__check_inputs()


    def __len__(self) -> int:
        """
        return the length of the dataset
        """
        return len(self.train_id)
    

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return the idx-th item of the dataset
        """
        label = self.txn.get(('gt_' + self.train_id[idx]).encode('ascii'))
        label_tensor = self.__read_lmdb(data = label, height = 64, width = 64)
        inputs = self.txn.get(('noisy_' + self.train_id[idx]).encode('ascii'))
        inputs_tensor = self.__read_lmdb(data = inputs, height = 64, width = 64)
        return inputs_tensor, label_tensor
    
    
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.train_lmdb_path.exists(), 'train_lmdb not found, please check the path'
        assert len(self.train_id) > 0, 'train_idx_list is empty, please check the path'
        self.env = lmdb.open(str(self.train_lmdb_path), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(buffers=True, write=False)
      
      
    def __read_lmdb(self, data: bytes, height: int, width: int) -> torch.Tensor:
        """
        read lmdb data and return tensor
        """
        return self.__to_tensor(np.frombuffer(data, dtype=np.float32).reshape((height, width, 4)))
    
    
    def __to_tensor(self, ndarray: np.ndarray) -> torch.Tensor:
        """
        convert ndarray to tensor
        """
        assert isinstance(ndarray, np.ndarray), 'ndarray should be numpy.ndarray, please check it'
        assert len(ndarray.shape) == 3, 'ndarray should be 3 dimension, please check it'
        assert ndarray.dtype == np.float32, 'ndarray should be np.float32, please check it'
        return torch.from_numpy(ndarray.astype(np.float32)).permute(2, 0, 1)
    

class Validset(Dataset):
    """
    Validset, read from valid_img and return noisy_tensor
    """
    def __init__(self):
        """
        root_path:
            root path of the project
        valid_path:
            path of the index file
        id_list:
            list of the index
        """
        self.root_path = Path(os.path.abspath(__file__)).parent.parent
        self.train_lmdb_path = self.root_path / 'IMX766' / 'train_data'
        self.id_list = open(self.root_path / 'IMX766' / 'train_data' / 'train_data_idx.txt', 'r').read().split()
        self.valid_id = self.id_list[-5:]
        self.__check_inputs()       

    
    def __len__(self) -> int:
        """
          return the length of the dataset
        """
        return len(self.valid_id)
      
       
    def __getitem__(self, idx) -> torch.Tensor:
        """
        return the idx-th item of the dataset
        """
        label = self.txn.get(('gt_' + self.valid_id[idx]).encode('ascii'))
        label_tensor = self.__read_lmdb(data = label, height = 64, width = 64)
        inputs = self.txn.get(('noisy_' + self.valid_id[idx]).encode('ascii'))
        inputs_tensor = self.__read_lmdb(data = inputs, height = 64, width = 64)
        return inputs_tensor, label_tensor
    
    
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert len(self.valid_id) > 0, 'valid_id is empty, please check the path'
        self.env = lmdb.open(str(self.train_lmdb_path), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(buffers=True, write=False)


    def __to_tensor(self, ndarray: np.ndarray) -> torch.Tensor:
        """
        convert ndarray to tensor
        """
        assert isinstance(ndarray, np.ndarray), 'ndarray should be numpy.ndarray, please check it'
        assert len(ndarray.shape) == 3, 'ndarray should be 3 dimension, please check it'
        return torch.from_numpy(ndarray.astype(np.float32)).permute(2, 0, 1)
    
    
    def __read_lmdb(self, data: bytes, height: int, width: int) -> torch.Tensor:
        """
        read lmdb data and return tensor
        """
        return self.__to_tensor(np.frombuffer(data, dtype=np.float32).reshape((height, width, 4)))
    
    
class Testset(Dataset):
    """
    Testset, read from test_img
    """
    def __init__(self):
        """
        root_path:
            root path of the project
        valid_path:
            path of the index file
        id_list:
            list of the index
        """
        self.root_path = Path(os.path.abspath(__file__)).parent.parent
        self.test_list = self.root_path / 'IMX766' / 'test_data'
        self.test_id = glob.glob(str(self.test_list) + '/*')
        print(self.test_id)
        self.__check_inputs()       

    
    def __len__(self) -> int:
        """
          return the length of the dataset
        """
        return len(self.test_id)
      
       
    def __getitem__(self, idx) -> torch.Tensor:
        """
        return the idx-th item of the dataset
        """
        return self.__read_tensor_from_raw(self.test_id[idx])
    
    
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert len(self.test_id) > 0, 'test_id is empty, please check the path'


    def __to_tensor(self, ndarray: np.ndarray) -> torch.Tensor:
        """
        convert ndarray to tensor
        """
        assert isinstance(ndarray, np.ndarray), 'ndarray should be numpy.ndarray, please check it'
        assert len(ndarray.shape) == 3, 'ndarray should be 3 dimension, please check it'
        return torch.from_numpy(ndarray.astype(np.float32)).permute(2, 0, 1)
    
    
    def __read_tensor_from_raw(self, input_path:str):
        feedraw= FeedRAW(input_path)
        raw_data, meta_data = feedraw.raw_data, feedraw.meta_data
        blc, wlc, _ = meta_data['black_level'], meta_data['white_level'], meta_data['bayer_pattern']
        split_noisy_rggb = Bayer2RGGB(raw_data)
        split_noisy_norm = normalization(split_noisy_rggb, black_level = blc, white_level = wlc)
        return self.__to_tensor(split_noisy_norm)
