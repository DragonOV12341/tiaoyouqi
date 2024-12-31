# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import subprocess
from internal import *
from model_manager import *
from global_status import *
    
def main_process(info : UserInputInfo) :
    # model_path = '/home/xushilong/tiaoyouqi/stub_code/VGG19.py'
    # logPath = '/home/xushilong/tiaoyouqi/stderr_output.txt'
    model_path = info.m_modelFilePath
    irPath = '/home/xushilong/tiaoyouqi/model_ir.mlir'
    codegenpath = '/home/xushilong/tiaoyouqi/codgendir'
    plat = 'dcu'
    print("reading model & convert into IR ...")
    GlobalStatus().set_runningStatus("Parsing Models")
    cmd = ["python","model_manager.py",model_path]

    Model,ModelInputs,RunModel = import_model(model_path) 
    with open(irPath,'w') as f :
        subprocess.call(cmd,stdout=f,stderr=f)
        print("convert model OK !")
    # analysis model operators
    mm = ModelManager(plat,codegenpath,RunModel,ModelInputs)
    mm.analysis(irPath)
    # codegen
    mm.codegen()
    # run model
    print("======== start run model =========")
    mm.test_e2e_time()


if __name__ == "__main__":
    modelPath = '/home/xushilong/tiaoyouqi/sample_model/resnet50.py'
    main_process(UserInputInfo(modelPath,'dcu'))