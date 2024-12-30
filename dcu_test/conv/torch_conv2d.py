import torch
import torch.nn as nn

import numpy as np
import datetime
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    # batch 输入通道 输出通道 输入形状 输出形状 卷积核大小 步长 填充大小 核内元素间隔
    [1, 64, 64, 112, 56, 3, 2, 1, 1],
    [1, 64, 128, 56, 28, 3, 2, 1, 1],
    [1, 128, 256, 28, 14, 3, 2, 1, 1],
    [1, 160, 160, 64, 64, 3, 1, 1, 1],
    [1, 240, 240, 32, 32, 3, 1, 1, 1],
    [1, 320, 320, 16, 16, 3, 1, 1, 1],
  ]

def test_conv2d(idx=0):
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
  model = conv_op(inC, outC, kernel_size, stride=stride, padding=padding, 
        dilation=dilation, bias=True, dtype=dtype, device=device)

  # print(model)
  # print("input shape:", dummy_input.shape)
  output = model(dummy_input)
  # print("output shape:", output.shape)

  from datetime import datetime
  res = []

  for j in range(10):
    start_time = datetime.now()
    
    output = model(dummy_input)

    torch.cuda.synchronize()
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds() * 1000
    # print(f"matmul datetime time: {elapsed_time:.2f} ms")
    res.append(elapsed_time)

  # print(f"10 test total time: {elapsed_time} ms")
  print(f"median time: {np.max(res)} ms")
  mid_time = round(np.max(res), 4)
  FLOPS = batch*outHnW*outHnW*outC*kernel_size*kernel_size*inC*2
  tflops = FLOPS / (mid_time / 1000) / 1e12
	# print("tflops: ", round(FLOPS, 4))
  print("TFLOPS: ", round(tflops, 4))
  # print(batch, outHnW, outHnW, outC, kernel_size, kernel_size, inC)

if torch.cuda.is_available():
	print("using", device)
	# test_conv2d(5)
	for i in range(6):
		print("---------------------------------------------------")
		print("conv2d test case:", i)
		test_conv2d(i)
else:
  print(device)