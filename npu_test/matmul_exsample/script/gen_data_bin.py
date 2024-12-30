import numpy as np
import os
from PIL import Image
import sys

shape_array = [
    [1, 1, 4096, 4096],
    [16, 20, 32, 20],
    [2, 20, 256, 768],
    [2, 20, 256, 1024],
    [128, 1, 128, 58],
    [128, 1, 58, 128],
]

def process(idx):
    try:
        img1 = np.random.randn(shape_array[idx][0], shape_array[idx][1], shape_array[idx][2]).astype("float16")
        img2 = np.random.randn(shape_array[idx][0], shape_array[idx][2], shape_array[idx][3]).astype("float16")
        output_name1 = "A" + str(idx) + ".bin"
        output_name2 = "B" + str(idx) + ".bin"
        img1.tofile(output_name1)
        img2.tofile(output_name2)
        print("A shape:", img1.shape)
        print("out file:", output_name1)
        print("B shape:", img2.shape)
        print("out file:", output_name2)
    except Exception as except_err:
        print(except_err)
        return 1
    else:
        return 0

if __name__ == "__main__":
    caseidx = 0
    if len(sys.argv) == 2:
        caseidx = int(sys.argv[1])
    process(caseidx)