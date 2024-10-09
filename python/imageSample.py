#################################################################
# noiseReduce Tensorflow Project 
# Copyright (C) 2024 Simeon Symeonidis
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#################################################################


# import libraries
import numpy as np
import random

def getDim(src, window_size, num_samples):
  size  = np.shape(src)
  max_x = size[0] - window_size[0] - 1
  max_y = size[1] - window_size[1] - 1
  dimIn = src.ndim
  if dimIn == 2:
    dim = (num_samples, window_size[0], window_size[1])
  else:
    dim = (num_samples, window_size[0], window_size[1], size[2])
  return (dim, max_x, max_y)

# sample function
def sampleRnd(src, window_size, num_samples):
  (dim, max_x, max_y) = getDim(src, window_size, num_samples)
  out = np.empty(dim, src.dtype)
  for i in range(num_samples):
    x = random.randrange(max_x)
    y = random.randrange(max_y)
    out[i,:] = src[x:x+window_size[0],y:y+window_size[1]]
  return out

# sample function
def sampleRnd2(src1, src2, window_size, num_samples):
  (dim, max_x, max_y) = getDim(src1, window_size, num_samples)
  out1 = np.empty(dim, src1.dtype)
  out2 = np.empty(dim, src2.dtype)
  for i in range(num_samples):
    x = random.randrange(max_x)
    y = random.randrange(max_y)
    out1[i,:] = src1[x:x+window_size[0],y:y+window_size[1]]
    out2[i,:] = src2[x:x+window_size[0],y:y+window_size[1]]
  return (out1, out2)

# command line interface
if __name__ == "__main__":
  import imageUtilsPIL
  import matplotlib.pyplot as plt
  import sys
  img = imageUtilsPIL.imageRead(sys.argv[1])
  x   = int(sys.argv[2])
  y   = int(sys.argv[3])
  out = sampleRnd(img, [x, y], 16)
  plt.figure()
  for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.imshow(out[i,:])
    plt.axis("off")
  plt.show()
