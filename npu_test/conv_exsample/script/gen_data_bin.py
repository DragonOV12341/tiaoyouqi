import numpy as np
import os
from PIL import Image
import sys

shape_array = [
    [1, 64, 112, 112],
    [1, 64, 56, 56],
    [1, 128, 28, 28],
    [1, 320, 64, 64],
    [1, 640, 32, 32],
    [1, 1280, 16, 16],
]

def process(idx):
    try:
        img = np.random.randn(*(shape_array[idx]))
        img = img.astype("float16")
        output_name = "input" + str(idx) + ".bin"
        img.tofile(output_name)
        print("shape:", img.shape)
        print("out file:", output_name)

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