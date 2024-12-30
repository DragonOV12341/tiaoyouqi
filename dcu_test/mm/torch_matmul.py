import torch
import time
import numpy as np # type: ignore

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

dtype = torch.float32

tcases = [
  [1, 1, 2048, 2048],
  [16, 20, 32, 20],
  [2, 20, 256, 768],
  [2, 20, 256, 1024],
  [128, 1, 128, 58],
  [128, 1, 58, 128]
]

def test_matmul_time(batch = 1, len_m = 1024, len_k = 1024, len_n = 1024):
  weight = torch.randn(batch, len_k, len_n).to(device).to(dtype)
  bias = torch.randn(batch, len_m, len_n).to(device).to(dtype)
  input = torch.randn(batch, len_m, len_k).to(device).to(dtype)
  output = torch.matmul(input, weight)
  
  # print(output.size(), output.device, output.dtype)

  from datetime import datetime
  res = []
  
  for i in range(20):
    # print("---------------------------------------------------")
    start_time = datetime.now()

    output = torch.matmul(input, weight)
    torch.cuda.synchronize()
    output = torch.add(output, bias)

    torch.cuda.synchronize()
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds() * 1000
    # print(f"matmul datetime time: {elapsed_time:.2f} ms")
    res.append(elapsed_time)

  
  # mid_time = round(np.median(res), 4)
  # if k == 2048:
  mid_time = round(np.max(res)/2+np.median(res)/2, 4)
  # print(f"mid time: {mid_time} ms")
  FLOPS = 2*batch*len_m*len_n*len_k + batch*len_m*len_n
  # FLOPS += len_m*len_n
  gflops = FLOPS / (mid_time / 1000) / 1e12

  print("TFLOPS: ", round(gflops, 4))
  # print("---------------------------------------------------")
  # output = output.to("cpu")
  # input = input.to("cpu")
  # weight = weight.to("cpu")

import argparse
parser = argparse.ArgumentParser(description='parse case index')
parser.add_argument('--idx', type=int, default=-1, help='case index')
args = parser.parse_args()

if torch.cuda.is_available():
  if(args.idx == -1):
    for i in range(6):
      batch = tcases[i][0]
      m = tcases[i][1]
      k = tcases[i][2]
      n = tcases[i][3]
      # print("test case:", i, "batch-m-k-n:", batch, m, k, n)
      print("---------------------------------------------------")
      print("matmul test case:", i)
      test_matmul_time(batch, m, k, n)
  else:
    if(args.idx >= len(tcases) or args.idx < 0):
      print(f"idx invalid, idx must between 0 and {len(tcases)-1}")
      exit()
    batch = tcases[args.idx][0]
    m = tcases[args.idx][1]
    k = tcases[args.idx][2]
    n = tcases[args.idx][3]
    # print("test case:", args.idx, "batch-m-k-n:", batch, m, k, n)
    print("matmul test case:", args.idx)
    test_matmul_time(batch, m, k, n)
else:
  print(device)