#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/10/25
# @Description: train class


import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
import logging
import importlib
from make_dataloader import Trainset, Validset
import argparse
from pathlib import Path
from utils.tools import reduce_mean, set_random_seed, set_config, calculate_psnr, calculate_ssim, Charbon_loss, simple_isp, save_valid_image
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.fx.graph_module import ObservedGraphModule
from torch.quantization import get_default_qconfig
import copy


class Trainer(object):
    """
    this is a class for training, including:
    1. load config from yaml
    2. set hyperparameters:
        .set dataloader
        .set loss
        .set optimizer
        .set lr_scheduler
        .set warmup
        .set tensorboard
        .set checkpoint
        .set logger
        .set ddp
        .set network
    3. run training
    
    usage:
        Trainer(train_config).run()
    """
    def __init__(self, train_config:str) -> None:
        """
        train_config:
            arguments from yaml
        """
        super().__init__()
        self.args = set_config(config = train_config)
        self.__check_inputs()
        self.__set_hyperparameters()
  
        
    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        set attribute
        """
        self.__dict__[__name] = __value

    
    def run(self) -> None:
        """
        run training
        """
        return self.__ddp_train_loop()
    

    def __check_inputs(self) -> None:
        """
        check inputs
        """
        assert self.args['criterion'] in ['l1', 'l2', 'charbon'], 'criterion should be in [l1, l2, charbon]'
        assert self.args['optimizer'] in ['adam', 'sgd'], 'optimizer should be in [adam, sgd]'
        assert self.args['lr_scheduler'] in ['cosine', 'step'], 'lr_scheduler should be in [cosine, step]'
        assert self.args['use_warm_up'] in [True, False], 'warm_up should be in [True, False]'
        assert self.args['use_tensorboard'] in [True, False], 'use_tensorboard should be in [True, False]'
        assert self.args['use_summarywriter'] in [True, False], 'use_summarywriter should be in [True, False]'
        assert self.args['use_logger'] in [True, False], 'use_logger should be in [True, False]'
        assert self.args['use_lr_scheduler'] in [True, False], 'use_lr_scheduler should be in [True, False]'
        

    def __set_hyperparameters(self) -> None:
        """
        set hyperparameter
        """
        self.__set_network()
        self.__set_ddp_config()
        self.__set_save_log()
        self.__set_tensorboard()
        self.__set_checkpoint()
       
        self.__set_dataloader()
        self.__set_loss()
        self.__set_optimizer()
        self.__set_lr_scheduler()
        self.__set_warmup()
   

    def __set_network(self) -> None:
        """
        set model
        """
        m = self.args['network']
        network = importlib.import_module(f'model_zoo.{m}')
        self.network = getattr(network, m)() if hasattr(network, m) else None
        
    
    def __set_save_log(self) -> None:
        """
        set save_log
        """
      
        def beijing(sec, what):
            import datetime
            beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
            return beijing_time.timetuple()
        
        if self.args['use_logger']:
            self.log_dir = self.args['log_dir']
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.args['valid_dir'], exist_ok=True)
            logging.getLogger().setLevel(logging.INFO)
            logging.Formatter.converter = beijing
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s',
                    level=logging.INFO,
                    filename=os.path.join(self.log_dir, self.args['network'] +'.log'),
                    filemode='w')
            logging.info(f'args: {self.args}')

           
    def __set_optimizer(self) -> None:
        """
        set optimizer
        """
        optimizer_dict = {

            'adam': optim.Adam(self.network.parameters(),
                                lr=self.args['learning_rate'],
                                betas=(0.9, 0.999),
                                weight_decay=self.args['weight_decay']),
            'sgd': optim.SGD(self.network.parameters(),
                                lr=self.args['learning_rate'],
                                weight_decay=self.args['weight_decay']),
            'adamw': optim.AdamW(self.network.parameters(),
                                lr=self.args['learning_rate'],
                                betas=(0.9, 0.999),
                                weight_decay=self.args['weight_decay']),
            'adagrad': optim.Adagrad(self.network.parameters(),
                                    lr=self.args['learning_rate'],
                                    weight_decay=self.args['weight_decay'])
        }
        self.optimizer = optimizer_dict.pop(self.args['optimizer'])
        return self.optimizer
        
        
    def __set_loss(self) -> None:
        """
        set loss
        """
        criterion_dict = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss(),
            'charbon': Charbon_loss()
        }
        self.criterion = criterion_dict.pop(self.args['criterion'])
        return self.criterion
          
        
    def __set_dataloader(self) -> None:
        """
        set dataloader
        """
        self.train_sampler, self.train_loader = self.__train_dataloader()
        self.val_loader = self.__val_dataloader()


    def __train_dataloader(self) -> None:
        """
        set train dataloader
        """
        train_sampler = DistributedSampler(Trainset())
        train_loader = torch.utils.data.DataLoader(Trainset(),
                                                    batch_size=self.args['train_batch_size'],
                                                    num_workers=self.args['train_num_workers'],
                                                    pin_memory=True,
                                                    sampler=train_sampler)
        return train_sampler, train_loader

        
    def __val_dataloader(self) -> None:
        """
        set val dataloader
        """
        return torch.utils.data.DataLoader(Validset(),
                                        batch_size=self.args['valid_batch_size'],
                                        num_workers=self.args['valid_num_workers'],
                                        pin_memory=True,
                                        shuffle=True)
        

    def __set_lr_scheduler(self) -> None:
        """
        set lr_scheduler
        """
        lr_scheduler_dict = {
            'cosine': optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.args['train_epochs'] * 2),
        }
        self.lr_scheduler = lr_scheduler_dict.pop(self.args['lr_scheduler'])
        return self.lr_scheduler
        
        
    def __set_warmup(self) -> None:
        """
        set warmup
        """
        if self.args['use_warm_up']:
            self.warmup =  GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=3, after_scheduler=self.lr_scheduler)
            
        
    def __set_tensorboard(self) -> None:
        """
        set tensorboard
        """
        if self.args['use_tensorboard']:
            self.writer = SummaryWriter(log_dir=self.args['tensorboard_dir'])
            os.makedirs(self.args['tensorboard_dir'], exist_ok=True)
            
            
    def __set_checkpoint(self) -> None:
        """
        set checkpoint
        """
        self.checkpoint_dir = self.args['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
                        
    def __set_ddp_config(self) -> None:
        """
        set ddp
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int)
        pars = parser.parse_args()
        torch.cuda.set_device(pars.local_rank)
        device=torch.device("cuda", pars.local_rank)
        dist.init_process_group(backend = 'nccl')
        self.network.to(device)
        self.network = DDP(self.network, device_ids = [pars.local_rank], output_device= pars.local_rank, find_unused_parameters=True)
        set_random_seed(self.args['seed'] + pars.local_rank)

        
    def __ddp_train_loop(self) -> None:
        """
        train loop
        """
        # if self.args['checkpoint'] is not None:

       
        logging.info(f"Restart training with ddp")
        self.global_step = 0
        self.best_loss = 1e6
        for epoch in tqdm(range(self.args['train_epochs'])):
            self.network.train()
            self.train_sampler.set_epoch(epoch)
            self.__ddp_train(epoch)
            self.__valid(epoch) if dist.get_rank() == 0 else None
            self.lr_scheduler.step()
            self.warmup.step()
            if self.args['use_tensorboard']:
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
  
  
    def __ddp_train(self, epoch:int) -> None:
        """
        train with ddp
        """
        ave_loss = list()
        for step, (images, labels) in enumerate(self.train_loader):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            self.optimizer.zero_grad()
            outputs = self.network(images)
            loss = self.criterion(outputs, labels)
            loss = reduce_mean(loss, dist.get_world_size())
            loss.backward()
            self.optimizer.step()
            ave_loss.append(loss.item())
            if dist.get_rank() == 0:
                if self.args['use_tensorboard']:
                    self.writer.add_scalar('train_loss', loss.item(), self.global_step)
                if self.global_step % self.args['print_step'] == 0:
                    logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, best_Loss: {:.4f}, Global_step:{}'.format(epoch, self.args['train_epochs'], step, len(self.train_loader), loss.item(), self.best_loss, self.global_step))
                self.global_step += 1
        ave_loss = (sum(ave_loss)/len(ave_loss))
        if self.best_loss > ave_loss:
            self.best_loss = ave_loss
            torch.save(self.network.module.state_dict() if hasattr(self.network, "module") else self.network.state_dict(),
                       os.path.join(self.checkpoint_dir, self.args['network'] + '_best_ckpt.pth'))
        if epoch == self.args['train_epochs'] - 1 and dist.get_rank() == 0:
            # torch.save(self.network.state_dict(), os.path.join(self.checkpoint_dir, self.args['network'] + '_last_ckpt.pth'))
            # logging.info(f'model has been saved in {self.checkpoint_dir}')
            if self.args['use_quant']:
                network_quant = self.__quant_fx()
                logging.info(f'model has been quantized and saved in {self.checkpoint_dir}, please use the quantized model for inference')
                self.__valid_quant(network_quant, epoch)
                
    
    @torch.no_grad()
    def __valid(self, epoch:int) -> None:
        """
        valid
        """
        self.network.eval()
        ave_psnr = list()
        ave_ssim = list()
        for step, (images, labels) in enumerate(self.val_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = self.network(images)
            psnr = calculate_psnr(outputs, labels)
            ssim = calculate_ssim(outputs, labels)
            ave_psnr.append(psnr)
            ave_ssim.append(ssim)
            noisy_rgb = simple_isp(images)
            clean_rgb = simple_isp(labels)
            nr_rgb = simple_isp(outputs)
            save_valid_image(self.args['valid_dir'] + '/' + f'{step}_output.png' , nr_rgb)
            save_valid_image(self.args['valid_dir'] + '/' + f'{step}_label.png', clean_rgb)
            save_valid_image(self.args['valid_dir'] + '/' +  f'{step}_input.png', noisy_rgb)

        ave_psnr = (sum(ave_psnr)/len(ave_psnr))
        ave_ssim = (sum(ave_ssim)/len(ave_ssim))
        if self.args['use_tensorboard']:
            self.writer.add_scalar('valid_PSNR', ave_psnr, epoch)
            self.writer.add_scalar('valid_SSIM', ave_ssim, epoch)
            
            
    @torch.no_grad()
    def __valid_quant(self, network_quant, epoch:int) -> None:
        """
        valid network_quant
        """
        network_quant.eval()
        ave_psnr = list()
        ave_ssim = list()
        for step, (images, labels) in enumerate(self.val_loader):
            outputs = network_quant(images)
            psnr = calculate_psnr(outputs, labels)
            ssim = calculate_ssim(outputs, labels)
            ave_psnr.append(psnr)
            ave_ssim.append(ssim)

        ave_psnr = (sum(ave_psnr)/len(ave_psnr))
        ave_ssim = (sum(ave_ssim)/len(ave_ssim))
        if self.args['use_tensorboard']:
            self.writer.add_scalar('valid_PSNR', ave_psnr, int(epoch-1))
            self.writer.add_scalar('valid_SSIM', ave_ssim, int(epoch-1))
        
        
    def __quant_fx(self) -> nn.Module:
        """
        quantize model with fx
        """
        @torch.no_grad()
        def calib_quant_model(model):
            assert isinstance(model, ObservedGraphModule), "model must be a perpared fx ObservedGraphModule."
            model.cpu()
            model.eval()
            for inputs, labels in self.val_loader:
                model(inputs)
        network_quant = copy.deepcopy(self.network.module if hasattr(self.network, "module") else self.network)
        qconfig = get_default_qconfig("fbgemm")
        qconfig_dict = {
            "": qconfig,
            # 'object_type': []
        }
        network_quant = prepare_fx(network_quant, qconfig_dict, example_inputs = (torch.randn(1, 4, 160, 160).cuda(), ))
        calib_quant_model(network_quant)
        network_quant = convert_fx(network_quant)
        torch.save(network_quant.state_dict(), os.path.join(self.checkpoint_dir, self.args['network'] + '_quant_ckpt.pth'))
        return network_quant
        

if __name__ == "__main__":
    root_path = Path(os.path.abspath(__file__)).parent.parent
    config_path = root_path / 'train_model' / 'train_config.yaml'
    train_model = Trainer(train_config=config_path)
    train_model.run()
    
                     


        
