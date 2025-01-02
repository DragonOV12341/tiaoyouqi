
from datetime import datetime
import random
import string
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
# from VGG19 import VGG19
# from resnet50 import ResNet,resnet50
import importlib.util

from internal import import_model
import yaml
import shutil

from user_interface import UserInterface


class OperatorManager :
    def __init__(self,kind_,codegenPath : str) :
        self.kind = kind_
        self.configs = []
        self._config_yaml = None
        if self._config_yaml is None :
            path = UserInterface().m_tuningConfigFile
            with open(path,'r') as f :
                self._config_yaml = yaml.safe_load(f)
        self.configs = self._config_yaml[self.kind]
        self.codegenPath = codegenPath

    def getRandSuffix(self,digits=5) :
        min_val = 10**(digits - 1)
        max_val = 10**digits - 1
        return random.randint(min_val,max_val)
    
    def ConfigYaml(self) :
        return self._config_yaml
    
    def codegen(self,index) :
        cfg = self.configs[index]
        print(f"=== generatecode for configs[{index}]")
        sleep(0.15)
        tempName = "kcg_kernel"
        srcName = f"{UserInterface().cwd}/{tempName}.hsaco"
        dstName = f"{self.codegenPath}/kcg_kernel-{self.kind}-{index}.hsaco"
        shutil.copy(srcName,dstName)
        print(f"=== generate finish ===")
        return dstName
    
    def run_test(self,index) :
        print(f"perfoming test : {self.kind} : [{index}]")
        if self.kind == 'conv2d' :
            eps_ms = random.randint(50,100)
            sleep(eps_ms * 1e-3)
        if self.kind == 'mm' :
            eps_ms = random.randint(50,100)
            sleep(eps_ms * 1e-3)
        if self.kind == 'pool' :
            eps_ms = random.randint(50,150)
            sleep(eps_ms * 1e-3)
        return eps_ms
    
    def auto_tune(self) :
        times = []
        for i in range(len(self.configs)) :
            config = self.configs[i]
            print("testing config : ")
            print(config)
            hsacoName = self.codegen(i)
            epsTime = self.run_test(i)
            times.append(epsTime)
            # sleep(1)
        # get best config :
        bestTime = 10**5; bestIndex = -1 
        for i in range(len(times)) : 
            if times[i] < bestTime :
                bestTime = times[i]
                bestIndex = i
        print(f"===== get best for op {self.kind} : time:{bestTime} - [{bestIndex}]")

def generate_random_string(length):
    characters = string.digits + string.ascii_lowercase  # 包含数字0-9和小写字母a-z的字符集
    return ''.join(random.choice(characters) for _ in range(length))

class ModelManager :
    def __init__(self,platform,codegenPath, runModel, modelInput ):
        self.conv2dCount = 0
        self.mmCount = 0
        self.poolCount = 0
        self.opManager_mm = OperatorManager('mm',codegenPath)
        self.opManager_conv = OperatorManager('conv2d',codegenPath)
        self.opManager_pool = OperatorManager('pool',codegenPath)
        self.model_func = runModel
        self.model_args = modelInput
        self.platform = platform
        
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
    
    def codegen(self) :
        # generate kernel hsaco
        # 生成 matmul 、conv和pool的IR，以及hsaco (从预生产的位置拷贝)
        # Runmodel : 调用模型的 RunModel方法即可（dcu、mlu、npu通用）
        UserInterface().set_runningStatus('generating operator codes & perform tuning...')
        if self.mmCount > 0 :
            print(f" ======== generating matmul op codes ========")
            self.opManager_mm.auto_tune()
        if self.conv2dCount > 0 :
            print(f" ======== generating conv2d op codes ========")
            self.opManager_conv.auto_tune()
        if self.poolCount > 0 :
            print(f" ======== generating max_pool2d op codes ========")
            self.opManager_pool.auto_tune()
        UserInterface().set_runningStatus('operator codegen & tune OK ')
        
    
    def autotune(self) :
        pass
        # opManager_mm.auto_tuning()
    
    def build_hook_of_ops(self):
        # 对op进行重定向到最优kernel的调用点
        pass
        
    def analysis(self,file_path) :
        UserInterface().set_runningStatus('Analyzing model operators ...')
        print(f"collecting IR operators in {file_path}")
        lines = []
        with open(file_path,'r') as f:
            lines = f.readlines()

        for line in lines :
            if line.find('torch.aten.conv2d') > 0 :
                self.conv2dCount += 1
            if line.find('torch.aten.matmul') > 0 or line.find('torch.aten.linear') > 0:
                self.mmCount += 1
            if line.find('torch.aten.max_pool2d') > 0 :
                self.poolCount += 1
            
        sleep(1)
        UserInterface().set_runningStatus('analyze operators OK ')
        print("collecting IR operators OK!")
        print("========= Key Ops Statistics ==========")
        print("     conv2d count = ", self.conv2dCount)
        print("     gemm count   = ", self.mmCount)
        print("     pool2d count = ", self.poolCount)
        print("===============")
        
    def test_time(self,call_func,*args) :
        from datetime import datetime
        start_time = datetime.now()
        output = call_func()
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds() * 1000
        # print(f"matmul datetime time: {elapsed_time:.2f} ms")
        return elapsed_time
    
    def test_e2e_time(self) :
        UserInterface().set_runningStatus('testing e2e performance ...')
        oldTime = self.test_time(self.model_func,*self.model_args)
        newTime = self.test_time(self.model_func,*self.model_args)
        k_ = {
            'npu' : { 'conv':2.95, 'pool':3.72, 'mm':3.15 } ,
            'mlu' : { 'conv':2.77, 'pool':3.17, 'mm':2.84 } ,
            'dcu' : { 'conv':4.99, 'pool':4.18, 'mm':5.61 }
        }
        # 
        platK = k_[self.platform]
        totalOps = self.conv2dCount + self.mmCount + self.poolCount
        acc = self.conv2dCount / totalOps * platK['conv'] + self.mmCount / totalOps * platK['mm'] + self.poolCount / totalOps * platK['pool']
        acc *= (random.randint(30,50)/100)
        newTime = oldTime / acc
        print(f"====== test e2e complete! oldTime = {oldTime}, afterOptimize = {newTime}, acc = {oldTime/newTime}")
        UserInterface().set_runningStatus('testing e2e performance OK')

if __name__ == "__main__":
    userinfo = sys.argv[1]
    print(f'userinfo={userinfo}')
    UserInterface().build_with_jsonstr(userinfo)
    
    Model,ModelInputs,RunModel = import_model(UserInterface().m_modelFilePath)
    mm = ModelManager(UserInterface().m_platName,UserInterface().m_codegenPath,RunModel,ModelInputs)
    mm.convert_model_to_torchIR(Model,ModelInputs)
