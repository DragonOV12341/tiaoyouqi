import torch
import torch.nn as nn 
import numpy as np
import onnx
# import onnxsim
import datetime
import os
import sys

from torch_npu.contrib import transfer_to_npu
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)
dtype = torch.float16

class conv_op(nn.Module):
  def __init__(self, inC, outC, kernel_size, stride=1, padding=0, 
      dilation=1, bias=True, dtype=None, device=None):
    super().__init__()
    self.conv1 = nn.Conv2d(inC, outC, kernel_size, stride=stride, padding=padding, 
      dilation=dilation, bias=True, dtype=dtype, device=device)
  def forward(self, x):
    y = self.conv1(x)
    return y

tcases = [
    [1, 64, 64, 112, 56, 3, 2, 1, 1],
    [1, 64, 128, 56, 28, 3, 2, 1, 1],
    [1, 128, 256, 28, 14, 3, 2, 1, 1],
    [1, 320, 320, 64, 64, 3, 1, 1, 1],
    [1, 640, 640, 32, 32, 3, 1, 1, 1],
    [1, 1280, 1280, 16, 16, 3, 1, 1, 1],
  ]

caseidx = 0
if len(sys.argv) == 2:
  caseidx = int(sys.argv[1])

tcase = tcases[caseidx]

batch = tcase[0]
inC = tcase[1]
outC = tcase[2]
inHnW = tcase[3]
outHnW = tcase[4]
kernel_size = tcase[5]
stride = tcase[6]
padding = tcase[7]
dilation = tcase[8]

dummy_input = torch.randn(batch, inC, inHnW, inHnW).to(device).to(dtype)
model = conv_op(inC, outC, kernel_size, stride=stride, padding=padding, 
      dilation=dilation, bias=True, dtype=dtype, device=device)

print(model)
print("input shape:", dummy_input.shape)
output = model(dummy_input)
print("output shape:", output.shape)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

from datetime import datetime
res = []

for j in range(10):
  start_time = datetime.now()
  
  output = model(dummy_input)

  end_event.record()
  torch.cuda.synchronize()
  end_time = datetime.now()
  elapsed_time = (end_time - start_time).total_seconds() * 1000
  # print(f"matmul datetime time: {elapsed_time:.2f} ms")
  res.append(elapsed_time)

elapsed_time = start_event.elapsed_time(end_event)
# print(f"10 test total time: {elapsed_time} ms")
print(f"median time: {np.median(res)} ms")

FLOPS = batch*outHnW*outHnW*outC*kernel_size*kernel_size*inC*2
tflops = FLOPS / (np.median(res) / 1000) / 1e12

print("conv2d TFLOPS: ", round(tflops, 3))

log = "TFLOPS of conv case" + str(caseidx) + ": " + str(round(tflops, 3))
log_path = "./logs_conv.txt"
with open(log_path, 'a', encoding='utf-8') as file:
    file.write(log + '\n')

# model_scripted = torch.jit.script(model) 
# model_trace = torch.jit.trace(model, dummy_input) 

onnx_path = "./conv0.onnx"
# om_path = "./conv0.om"
# js_path = "./fusion_result.json"
# if os.path.exists(onnx_path):
#   os.remove(onnx_path)
# if os.path.exists(om_path):
#   os.remove(onnx_path)
# if os.path.exists(js_path):
#   os.remove(onnx_path)

torch.onnx.export(model, # model being run /model_trace
      dummy_input,       # model input (or a tuple for multiple inputs) 
      onnx_path,       # where to save the model  
      export_params=True,  # store the trained parameter weights inside the model file 
      opset_version=13,    # the ONNX version to export the model to 
      do_constant_folding=True,  # whether to execute constant folding for optimization 
      input_names = ['modelInput'],   # the model's input names 
      output_names = ['modelOutput'], # the model's output names 
      )

input_str = "\"modelInput:" + str(batch) + "," + str(inC) + "," + str(inHnW) + "," + str(inHnW) + "\""
atc_cli = "atc --model=conv0.onnx --framework=5 --output=conv0 --input_shape=" + input_str + " --soc_version=Ascend910B"
print(atc_cli)
os.system(atc_cli)
