
import random
from time import sleep
from typing import List

import torch
import torch.nn as nn
import torch_mlir
from torch.export import Dim
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import (
    make_boxed_compiler,
    get_aot_graph_name,
    set_model_name,
)

import gc
import sys

# from torch_mlir import torchscript
from torch_mlir import fx

from torch_mlir.compiler_utils import run_pipeline_with_repro_report
import torch_mlir.compiler_utils
from VGG19 import VGG19
from resnet50 import ResNet,resnet50
import importlib.util

from internal import import_model
import yaml



    

class OperatorManager :
    def __init__(self,kind_ : str) :
        self.kind = kind_
        self.configs = []
        self._config_yaml = None
        if self._config_yaml is None :
            path = '/home/xushilong/tiaoyouqi/stub_code/tune_configs.yaml'
            with open(path,'r') as f :
                self._config_yaml = yaml.safe_load(f)
        self.configs = self._config_yaml[self.kind]
        
    def ConfigYaml(self) :
        return self._config_yaml
    
    def codegen(self) :
        pass
    
    def test_config(self, cfg):
        print("=== testing cfg : ")
        print(cfg)
        elapsed_time = random(20,30)
        
        pass
    def auto_tuning(self) :
        minTime = 0
        for config in self.configs :
            print("testing config : ")
            print(config)
            sleep(1)
        print("===== get best config for op : ", self.configs[random(0,len(self.configs))])


class ModelManager :
    def __init__(self):
        self.conv2dCount = 0
        self.mmCount = 0
        self.poolCount = 0
        pass
        
    def convert_model_to_torchIR(self,model,inputs):
        # model definition
        # model = self.Model
        # input args
        # inputs = self.ModelInputs
        # convert to torch IR
        m = fx.export_and_import(model,inputs)
        m.dump()
        from torch_mlir.compiler_utils import OutputType
        # mm = torch_mlir.compiler_utils.lower_mlir_module(module=m,output_type=OutputType.LINALG_ON_TENSORS,verbose=False)
    
    def codegen(self,model) :
        # generate kernel hsaco
        # 生成 matmul 、conv和pool的IR，以及hsaco (从预生产的位置拷贝)
        # Runmodel : 调用模型的 RunModel方法即可（dcu、mlu、npu通用）
        for i in range(self.mmCount) :
            print(f" ======== generating matmul op codes. [{i}/{self.mmCount}] ========")
            sleep(1)
        for i in range(self.conv2dCount) :
            print(f" ======== generating conv2d codes. [{i}/{self.conv2dCount}] ========")
            sleep(1)
        for i in range(self.poolCount) :
            print(f" ======== generating pool codes. current op[{i}/{self.poolCount}] ========")
            sleep(1)
        pass
    
    def autotune(self) :
        opManager_mm = OperatorManager('mm')
        opManager_conv = OperatorManager('conv2d')
        opManager_pool = OperatorManager('maxpool')
        opManager_mm.auto_tuning()
    
    def build_hook_of_ops(self):
        # 对op进行重定向到最优kernel的调用点
        pass
        
    def analysis(self,file_path) :
        print("collecting IR operators ...")
        lines = []
        with open(file_path,'r') as f:
            lines = f.readlines()

        for line in lines :
            if line.find('torch.aten.conv2d') > 0 :
                self.conv2dCount+=1
            if line.find('torch.aten.linear') > 0 :
                self.mmCount+=1
            if line.find('torch.aten.max_pool2d') > 0 :
                self.poolCount+=1
        sleep(1)
        print("collecting IR operators OK!")
        print("========= Key Ops Statistics ==========")
        print("     conv2d count = ", self.conv2dCount)
        print("     gemm count   = ", self.mmCount)
        print("     pool2d count = ", self.poolCount)



if __name__ == "__main__":
    model_path = sys.argv[1]
    mm = ModelManager()
    Model,ModelInputs,RunModel = import_model(model_path)
    mm.convert_model_to_torchIR(Model,ModelInputs)
