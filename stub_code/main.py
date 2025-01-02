# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import subprocess
from internal import *
from model_manager import *
from user_interface import *
    
def main_process(info : UserInterface) :
    model_path = '/home/xushilong/tiaoyouqi/stub_code/VGG19.py'
    logPath = '/home/xushilong/tiaoyouqi/stderr_output.txt'
    model_path = info.m_modelFilePath
    outirPath = info.m_tempPath + '/model_ir.mlir'
    codegenpath = info.m_codegenPath
    plat = info.m_platName
    userinfo = UserInterface().dumpJsonString()
    print("reading model & convert into IR ...")
    UserInterface().set_runningStatus("Parsing Models")
    cmd = ["python","model_manager.py",userinfo]

    Model,ModelInputs,RunModel = import_model(model_path) 
    with open(outirPath,'w') as f :
        subprocess.call(cmd,stdout=f,stderr=f)
        print("convert model OK !")
    # analysis model operators
    mm = ModelManager(plat,codegenpath,RunModel,ModelInputs)
    mm.analysis(outirPath)
    # codegen
    mm.codegen()
    # run model
    print("======== start run model =========")
    mm.test_e2e_time()


if __name__ == "__main__":
    modelPath = '/home/xushilong/tiaoyouqi/sample_model/resnet50.py'  # 用户输入的 模型位置
    codegenPath = '/home/xushilong/tiaoyouqi/codgendir' # 用户指定的 算子输出目录
    tempPath = '/home/xushilong/tiaoyouqi/temp' # 用户指定的 临时文件缓存目录
    UserInterface().cwd = '/home/xushilong/tiaoyouqi/stub_code' # 当前工作目录，跟存放 tune_configs_xxx.yaml的路径一致
    UserInterface().set_modelFilePath(modelPath)
    UserInterface().set_platform('dcu')
    UserInterface().set_codegenPath(codegenPath)
    UserInterface().set_tempPath(tempPath)
    UserInterface().set_tuningConfigFile('standard')
    main_process(UserInterface())