import torch
import torch.nn as nn
import torch_mlu

import numpy as np
import datetime
import sys

device = torch.device("mlu" if torch.mlu.is_available() else "cpu")
dtype = torch.float32

class maxpool_op(nn.Module):
  def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
    super().__init__()
    self.pool1 = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)
  def forward(self, x):
    y = self.pool1(x)
    return y

class avgpool_op(nn.Module):
  def __init__(self, kernel_size, stride=1, padding=0):
    super().__init__()
    # torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
    self.pool1 = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
  def forward(self, x):
    y = self.pool1(x)
    return y

tcases = [
    #batch 输入通道 输出通道 输入形状 输出形状 卷积核大小 步长 填充大小 核内元素间隔
    [1, 96, 96, 55, 27, 3, 2, 0, 1],
    [1, 256, 256, 27, 13, 3, 2, 0, 1],
    [1, 256, 256, 13, 6, 3, 2, 0, 1],
    [1, 64, 64, 112, 56, 3, 2, 0, 1],
    [1, 2048, 2048, 7, 1, 7, 1, 0, 1],
    [1, 256, 256, 20, 20, 5, 1, 2, 1],
  ]

def test_pool2d(idx=0):
  tcase = tcases[idx]

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

  model = maxpool_op(kernel_size, stride=stride, padding=padding, 
        dilation=dilation)
  if(idx == 4):
    model = avgpool_op(kernel_size, stride=stride, padding=padding)

  # print(model)
  # print("input shape:", dummy_input.shape)
  output = model(dummy_input)
  # print("output shape:", output.shape)

  from datetime import datetime
  res = []

  for j in range(10):
    start_time = datetime.now()
    
    output = model(dummy_input)

    torch.mlu.synchronize()
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds() * 1000
    # print(f"matmul datetime time: {elapsed_time:.2f} ms")
    res.append(elapsed_time)

  # print(f"10 test total time: {elapsed_time} ms")
  # print(f"median time: {np.median(res)} ms")
  mid_time = round(np.max(res), 4)
  FLOPS = batch*outHnW*outHnW*outC*kernel_size*kernel_size*inC
  tflops = FLOPS / (mid_time / 1000) / 1e12

  print("TFLOPS: ", round(tflops, 4))

if torch.mlu.is_available():
  print("using", device)
  for i in range(6):
    print("---------------------------------------------------")
    print("pool2d test case:", i)
    test_pool2d(i)
else:
  print(device)