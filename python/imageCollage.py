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
import imageUtilsPIL
import fileUtils
import numpy as np
import math

# create split image
def gen(img1, img2):
  size  = np.shape(img1)
  x     = math.floor(size[1] / 2)
  out   = np.empty(size, 'uint8')
  out[:,:x,:] = img1[:,:x,:]
  out[:,x:,:] = img2[:,x:,:]
  return out
  
# command line interface
if __name__ == "__main__":
  import sys
  import ntpath
  import os
  import fileUtils
  import imageUtilsPIL
  files  = fileUtils.getFiles(sys.argv[1])
  for file in files:
    head, tail  = ntpath.split(file) 
    file1       = sys.argv[2] + tail
    file2       = sys.argv[3] + tail
    img1        = imageUtilsPIL.imageRead(file)    
    img2        = imageUtilsPIL.imageRead(file1)
    out         = gen(img1, img2)
    imageUtilsPIL.imageWrite(file2, out)    
