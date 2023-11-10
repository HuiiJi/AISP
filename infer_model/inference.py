#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/10/30
# @Description: inference class


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tools import set_config, time_cost_decorator, set_random_seed
import torch
import yaml
from typing import Dict, Any, Tuple, Union, List
from pathlib import Path
import importlib
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import copy
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx


class Inferencer():
    """
    Inference model
    
    usage:
            model = Inferencer(infer_config)
            output = model(inputs)
    """
    def __init__(self, infer_config:str) -> None:
        """
        infer_config:
            path of the model
        """
        self.args = set_config(config = infer_config)
        self.__check_inputs()
        self.__set_network()


    def __call__(self, inputs:torch.Tensor):
        """
        inputs:
            input tensor
        """
        forward_engine_dict = {
            'trt' : self.__inference_tensorrt,
            'torch': self.__inference_torch,
            'onnx': self.__inference_onnx
        }
        return forward_engine_dict[self.args['forward_engine']](inputs)
    
    
    def __check_inputs(self):
        """
        check inputs
        """
        assert self.args['ckpt_path'] , 'ckpt_path must be set'
        assert self.args['forward_engine'] in ['torch', 'trt', 'onnx'], 'forward_engine must be set'

        if self.args['forward_engine'] != 'torch':
            dir_path = os.path.abspath(os.path.dirname(__file__))
            self.onnx_dir = f'{dir_path}/onnx'
            self.tensorrt_dir = f'{dir_path}/tensorrt'
            os.makedirs(self.tensorrt_dir, exist_ok=True)
            os.makedirs(self.onnx_dir, exist_ok=True)
    
   
    def __set_network(self) -> None:
        """
        set model
        """
        m = self.args['network']
        network_class = importlib.import_module(f'model_zoo.{m}')
        network = getattr(network_class, m)() if hasattr(network_class, m) else None
        ckpt = torch.load(self.args['ckpt_path'])
        if self.args['use_qtorch']:
            qconfig = get_default_qconfig("fbgemm")
            qconfig_dict = {
                "": qconfig,
                # 'object_type': []
            }
            self.network = copy.deepcopy(network)
            self.network = prepare_fx(self.network, qconfig_dict, example_inputs = (torch.randn(1, 4, 128, 128),))
            self.network = convert_fx(self.network)
            self.network.load_state_dict(ckpt)
        else:
            self.network = network
            self.network.load_state_dict(ckpt)
    
    
    @time_cost_decorator
    @torch.no_grad()
    def __inference_torch(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inference pytorch model
        """
        device = inputs.device
        self.network.to(device)
        self.network.eval()
        return self.network(inputs).detach().cpu().numpy()
    

    @time_cost_decorator
    def __inference_onnx(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inference onnx model
        """
        onnx_path = self.__torch2onnx(input_shape=inputs.shape) if self.args['onnx_path'] is None else self.args['onnx_path']
        providers_option = ['CUDAExecutionProvider']
        onnx_engine = onnxruntime.InferenceSession(onnx_path, providers = providers_option)
        output = onnx_engine.run(None, {onnx_engine.get_inputs()[0].name: inputs.cpu().numpy()})[0]
        return output
    
    
    @time_cost_decorator
    def __inference_tensorrt(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inference tensorrt model
        """
        if self.args['tensorrt_path'] is None:
            onnx_path = self.args['onnx_path'] if self.args['onnx_path'] else self.__torch2onnx(self.args['input_shape'])[1]
            tensorrt_path = os.path.join(self.tensorrt_dir, self.args['network'] + '.engine')
            trt_engine = self.__onnx2tensorrt(onnx_path = onnx_path, tensorrt_path = tensorrt_path)
        else:
            trt_logger = trt.Logger(trt.Logger.ERROR)
            with open(self.args['tensorrt_path'], 'rb') as f:
                with trt.Runtime(trt_logger) as runtime:
                    trt_engine = runtime.deserialize_cuda_engine(f.read())
        input_bufs, output_bufs, bindings, stream = self.__allocate_buffer(trt_engine, input_shape = inputs.shape)
        context = self.__execute_context(trt_engine, input_shape=inputs.shape)
        input_bufs[0].host = inputs.cpu().numpy()
        cuda.memcpy_htod_async(
            input_bufs[0].device,
            input_bufs[0].host,
            stream
        )
        context.execute_async_v2(
            bindings=bindings,
            stream_handle=stream.handle
        )
        cuda.memcpy_dtoh_async(
            output_bufs[0].host,
            output_bufs[0].device,
            stream
        )
        stream.synchronize()
        return output_bufs[0].host.reshape(inputs.shape)
   
   
    def __allocate_buffer(self, engine: trt.ICudaEngine, input_shape: Tuple):
        """
        """
        class HostDeviceMem(object):
            def __init__(self, host_mem, device_mem):
                self.host = host_mem
                self.device = device_mem

            def __str__(self):
                return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

            def __repr__(self):
                return self.__str__()
        
        binding_names = []
        for idx in range(100):
            bn = engine.get_binding_name(idx)
            if bn:
                binding_names.append(bn)
            else:
                break

        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in engine:
            dims = engine.get_binding_shape(binding)
            if dims[-1] == -1:
                dims = input_shape
            size = trt.volume(dims) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream
    
    
    def __execute_context(self, engine: trt.ICudaEngine, input_shape: Tuple):
        """
        set input shape and execute context
        """
        with engine.create_execution_context() as context:
            context.active_optimization_profile = 0
            origin_inputshape=context.get_binding_shape(0)
            if (origin_inputshape[-1]==-1):
                origin_inputshape = input_shape
                context.set_binding_shape(0,(origin_inputshape))
        return context
                    
        
    def __torch2onnx(self, input_shape: tuple):
        """
        convert pytorch model to onnx model
        
        Args:
            model: pytorch model
            input_shape: input shape of model
        """
       
        def simplify_onnx(onnx_path: str, sim_onnx_path: str, use_netron:bool = False):
            """
            simplify onnx model

            Args:
                onnx_path: onnx model path
                output_path: output path of simplified onnx model
            """
            import onnx
            import netron
            from onnxsim import simplify
            onnx_model = onnx.load(onnx_path)
            model_simp, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simp, sim_onnx_path)
            onnx.checker.check_model(onnx.load(sim_onnx_path))
            print('onnx model is simplified and saved in {}'.format(sim_onnx_path))
            netron.start(sim_onnx_path) if use_netron else None
               
        x = torch.randn(input_shape)
        onnx_path = os.path.join(self.onnx_dir, self.args['network'] + '.onnx')
        sim_onnx_path = onnx_path.replace('.onnx', '_simplify.onnx')
        torch.onnx.export(self.network, 
                          x, 
                          onnx_path, 
                          verbose=False, 
                          opset_version=13, 
                          input_names=['inputs'], 
                          output_names=['outputs'],
                          dynamic_axes={'inputs': {0: 'batch_size', 2: 'height', 3: 'width'},
                                        'outputs': {0: 'batch_size', 2: 'height', 3: 'width'}})
        simplify_onnx(onnx_path = onnx_path, sim_onnx_path = sim_onnx_path)
        return sim_onnx_path
        
        
    def __onnx2tensorrt(self, onnx_path: str, tensorrt_path: str):
        """
        convert onnx model to tensorrt model
        
        Args:
            onnx_path: onnx model path
            tensorrt_path: output path of tensorrt model
            fp16_mode: fp16 mode
        """
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = 1 
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            config.set_flag(trt.BuilderFlag.FP16)
     
            
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):        
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))     
                    return None
            
            profile = builder.create_optimization_profile()
            profile.set_shape("inputs", (1,4,64,64), (1,4,128,128), (1,4,2560,2560))
            config.add_optimization_profile(profile)
            engine = builder.build_engine(network, config)
            with open(tensorrt_path, "wb") as f:
                f.write(engine.serialize()) 
            print('convert to tensorrt model successfully, saved in {}'.format(tensorrt_path))
            return engine
        


if __name__ == '__main__':
    root_path = Path(os.path.abspath(__file__)).parent.parent.parent
    config_path = root_path / 'Denoise' / 'infer_model' / 'infer_config.yaml'
    model = Inferencer(infer_config=config_path)
    set_random_seed(2023)
    x = torch.randn((1, 4, 128, 128)).cuda()
    output = model(x)
    print(output.shape, output.mean(), output.max(), output.min())


    
    