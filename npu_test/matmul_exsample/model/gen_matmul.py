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

class matmul_op(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x, y):
    z = torch.matmul(x, y)
    return z

# batchsize, M, K, N 
tcases = [
    [1, 1, 4096, 4096],
    [16, 20, 32, 20],
    [2, 20, 256, 768],
    [2, 20, 256, 1024],
    [128, 1, 128, 58],
    [128, 1, 58, 128]
  ]

caseidx = 0
if len(sys.argv) == 2:
  caseidx = int(sys.argv[1])

tcase = tcases[caseidx]

batch = tcase[0]
M = tcase[1]
K = tcase[2]
N = tcase[3]

dummy_A = torch.randn(batch, M, K).to(device).to(dtype)
dummy_B = torch.randn(batch, K, N).to(device).to(dtype)

model = matmul_op()

print(model)
print("batched matmul inputs shape:", dummy_A.shape, dummy_B.shape)
output = model(dummy_A, dummy_B)
print("output shape:", output.shape)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

from datetime import datetime
res = []

for j in range(10):
  start_time = datetime.now()
  
  output = model(dummy_A, dummy_B)

  end_event.record()
  torch.cuda.synchronize()
  end_time = datetime.now()
  elapsed_time = (end_time - start_time).total_seconds() * 1000
  # print(f"matmul datetime time: {elapsed_time:.2f} ms")
  res.append(elapsed_time)

elapsed_time = start_event.elapsed_time(end_event)
# print(f"10 test total time: {elapsed_time} ms")
print(f"median time: {np.median(res)} ms")

FLOPS = batch*M*K*N*2
tflops = FLOPS / (np.median(res) / 1000) / 1e12

print("matmul TFLOPS: ", round(tflops, 4))

log = "TFLOPS of matmul case" + str(caseidx) + ": " + str(round(tflops, 3))
log_path = "./logs_matmul.txt"
with open(log_path, 'a', encoding='utf-8') as file:
    file.write(log + '\n')

onnx_path = "./matmul0.onnx"

torch.onnx.export(model, # model being run /model_trace
      (dummy_A, dummy_B),       # model input (or a tuple for multiple inputs) 
      onnx_path,       # where to save the model  
      export_params=True,  # store the trained parameter weights inside the model file 
      opset_version=13,    # the ONNX version to export the model to 
      do_constant_folding=True,  # whether to execute constant folding for optimization 
      input_names = ['A', 'B'],   # the model's input names 
      output_names = ['C'], # the model's output names 
      )

# input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2
input_str = "\"A:" + str(batch) + "," + str(M) + "," + str(K) + ";B:" + \
    str(batch) + "," + str(K) + "," + str(N) + "\""
atc_cli = "atc --model=matmul0.onnx --framework=5 --output=matmul0 --input_shape=" + input_str + " --soc_version=Ascend910B"

print(atc_cli)
os.system(atc_cli)
