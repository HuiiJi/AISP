#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/10/15
# @Description: calibration class


import os
import numpy as np
from scipy.optimize import leastsq
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import yaml
import sys
from tools import FeedRAW, normalization, inv_normalization



class Noise_Calibration(object):
    """
    This class is used to calibrate noise profile from raw images.
    
    description:
        1. get black level and white level from raw images
        2. get noise profile from raw images

    step:
        1. read raw image
        2. get black level and white level
        3. get noise profile
        
    usage:
        calibrator_params = Noise_Calibration().run()
    """
    def __init__(self) -> None:
        super().__init__()
        """
        root_path: 
            the root path of the project
        blc:
            black level
        noise_model:
            noise model, should be in ['pg_model', 'pg_tl_model']
        """
        self.root_path = Path(os.path.abspath(__file__)).parent.parent
        self.blc = self.__get_black_level(root_path = self.root_path)
        self.wlc = 1023
        self.noise_model = 'pg_model'
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
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f'{self.root_path} is not exist, please check it')
        if not isinstance(self.root_path, Path) and isinstance(self.root_path, str):
            self.root_path = Path(self.root_path)
        if self.root_path not in sys.path:
            sys.path.append(self.root_path)
        assert self.noise_model in ['pg_model', 'pg_tl_model'], f'noise_model should be in [pg_model, pg_tl_model], please check it'
        os.makedirs(self.root_path / 'IMX766'/ 'noise_params', exist_ok=True)
     
    
    def run(self) -> Dict[str, np.ndarray]:
        """
        run the calibration, and return the calibrated noise profile
        """
        __noise_model_dict__ = {
            'pg_model': self.__calibrate_pg_model(root_path = self.root_path, blc = self.blc, wlc = self.wlc),
            'pg_tl_model': self.__calibrate_pg_tl_model(root_path = self.root_path, blc = self.blc, wlc = self.wlc)
            
        }
        return __noise_model_dict__.pop(self.noise_model)
    
    
    def __plt_show(self, image_id: str, mu: np.ndarray, var: np.ndarray, plsq: np.ndarray, save_path:Union[str, Path]) -> None:
        """
        plot the mu and var, and the fitted line
        
        Args:
            image_id:
                saved image name
            mu: 
                mean of the image
            var: 
                variance of the image
            plsq: 
                fitted line
            save_path: 
                saved path
        """
        assert isinstance(save_path, (str, Path)), f'save_path should be str or Path, but got {type(save_path)}'
        if isinstance(save_path, str):
            save_path = Path(save_path)
        assert save_path.exists(), f'{save_path} is not exist, please check it'
        assert isinstance(image_id, str), f'image_id should be str, but got {type(image_id)}'
        assert mu.shape == var.shape, f'mu and var should have the same shape, but got {mu.shape} and {var.shape}'
        plt.figure(figsize=(10, 10))
        plt.title(f'{image_id}')
        plt.xlabel('mean')
        plt.ylabel('var')
        plt.scatter(mu, var, color='b', label='data')
        plt.plot(mu, plsq[0] * mu + plsq[1], color='r', label=f'Fitted line')
        plt.legend()
        plt.savefig(save_path / f'calibrate_{image_id}.png')
        plt.close()
        
        
    def __error(self, p, mu, var)->np.ndarray:
        """
        error function for leastsq
        
        Args:
            p:
                fitted parameters, in shape (2,)
            mu:
                mean of the image
            var:
                variance of the image
        """
        assert p.shape == (2,), f'p should be in shape (2,), but got {p.shape}'
        assert mu.shape == var.shape, f'mu and var should have the same shape, but got {mu.shape} and {var.shape}'
        return (p[0] * mu + p[1]) - var
    
    
    def __get_roi_list(self) -> List[List[int]]:
        """
        get the roi list
        """
        roi_list = [[1200, 1350, 1400, 1450], [1200, 1600, 1400, 1700], [1200, 1850, 1400, 1950], [1200, 2150, 1400, 2250],
                [1200, 2400, 1400, 2500], [1200, 2650, 1400, 2750], [1200, 2900, 1400, 3000], [1200, 3150, 1400, 3250],
                [1200, 3400, 1400, 3500], [1200, 3650, 1400, 3750]]
        return roi_list
    
    
    def __get_black_level(self, root_path:str) -> float:
        """
        get the black level, the black level is the mean of the black image
        
        Args:
            root_path:
                root path of the project
        """
        black_img_list = sorted(os.listdir(root_path / 'IMX766' / 'black_img'))
        img_stack = [FeedRAW(root_path  / 'IMX766'/ 'black_img' / img).raw_data for img in black_img_list]
        black_level = float(np.stack(img_stack, axis=0).mean())
        return black_level
     

    def __calibrate_pg_model(self, root_path:str, blc:float, wlc:float, sensor_config:str = 'IMX766') -> None:
        """ 
        get the noise profile from the roi
        
        Args:
            root_path:
                root path of the project
            blc:
                black level, default 0
            wlc:
                white level, default 1023
            sensor_config:
                sensor config, default IMX766
        """
        assert sensor_config in ['IMX766', 'IMX766TL'], f'sensor_config should be in [IMX766, IMX766TL], please check it'
        assert isinstance(root_path, Path), f'root_path should be Path type, please check it'
        calibration_img_list = sorted(os.listdir(root_path / 'IMX766'/ 'calibrate_img'))
        img_stack = [FeedRAW(root_path / 'IMX766'/ 'calibrate_img' / img).raw_data for img in calibration_img_list]
        img_stack_id = [(os.path.basename(img).split('.')[0]) for img in calibration_img_list]
        img_stack_normalization = [normalization(inputs = img, black_level = blc, white_level = wlc) for img in img_stack]
        roi_list = self.__get_roi_list()
        shot_noise_list = list()
        read_noise_list = list()
        iso_select = list()
        sensor_profile = dict()
 
        for image, image_id in zip(img_stack_normalization, img_stack_id):
            mean_list = list()
            var_list = list()
            for roi in roi_list:
                region = image[roi[0]:roi[2], roi[1]:roi[3]]
                mean_list.append(region.mean())
                var_list.append(region.var())
            mu = np.asarray(mean_list)
            var = np.asarray(var_list)
            del mean_list, var_list
            plsq = leastsq(self.__error, [0.1, 0.01], args=(mu, var))[0]
            noise_params_path = self.root_path / 'IMX766' /'noise_params'
            self.__plt_show(image_id, mu, var, plsq, noise_params_path)
            shot_noise_list.append(plsq[0])
            read_noise_list.append(abs(plsq[1]))
            iso_select.append(int(image_id.split('_')[-1]))
            sensor_profile[f'{image_id}'] = f'shot_noise:{plsq[0]}', f'read_noise:{abs(plsq[1])}'
            
        k_shot =  np.polyfit(iso_select, shot_noise_list, 1).tolist()
        b_read =  np.polyfit(iso_select, read_noise_list, 2).tolist()
        sensor_profile['k_shot'] = k_shot
        sensor_profile['b_read'] = b_read
        yaml.dump(sensor_profile, open(noise_params_path / f'sensor_profile_{sensor_config}.yaml', 'w'))
        print(f' - successfully - noise_profile_{sensor_config}.yaml has been saved in {noise_params_path}')
        return (k_shot, b_read)
    
    
    def __calibrate_pg_tl_model(self, root_path:str, blc:float, wlc:float, sensor_config:str = 'IMX766') -> None:
        """
        will be added
        """
        pass



if __name__ == '__main__':
    Noise_Calibration().run()
pass
  
