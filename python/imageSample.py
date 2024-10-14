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
import math
import random

# calcuate number of fixed samples
def getSampleCount(src, window_size, skip_size):
  size   = np.shape(src)
  out_0  = math.floor((size[0] - window_size[0]) / skip_size[0]) + 1
  out_1  = math.floor((size[1] - window_size[1]) / skip_size[1]) + 1
  return out_0, out_1

# get dimensions of output array
def getDim(src, window_size, num_samples):
  size  = np.shape(src)
  max_x = size[0] - window_size[0] - 1
  max_y = size[1] - window_size[1] - 1
  dimIn = src.ndim
  if dimIn == 2:
    dim = (num_samples, window_size[0], window_size[1])
  else:
    dim = (num_samples, window_size[0], window_size[1], size[2])
  return dim, max_x, max_y

# fixed sample function
def sampleFixed(src, window_size, skip_size=(0,0)):
  if skip_size[0] == 0 or skip_size[1] == 0:
    skip_size      = window_size
  (size_x, size_y) = getSampleCount(src, window_size, skip_size)
  dim  = getDim(src, window_size, size_x * size_y)[0]
  out  = np.empty(dim, src.dtype)
  idx  = 0
  for i in range(size_x):
    y  = i * skip_size[0]
    for j in range(size_y):
      x           = j * skip_size[1]
      out[idx,:]  = src[y:y+window_size[0],x:x+window_size[1]]
      idx         = idx + 1
  return out

# random sample function
def sampleRnd(src, window_size, num_samples):
  (dim, max_x, max_y) = getDim(src, window_size, num_samples)
  out = np.empty(dim, src.dtype)
  for i in range(num_samples):
    x = random.randrange(max_x)
    y = random.randrange(max_y)
    out[i,:] = src[x:x+window_size[0],y:y+window_size[1]]
  return out

# random sample function
def sampleRnd2(src1, src2, window_size, num_samples):
  (dim, max_x, max_y) = getDim(src1, window_size, num_samples)
  out1 = np.empty(dim, src1.dtype)
  out2 = np.empty(dim, src2.dtype)
  for i in range(num_samples):
    x = random.randrange(max_x)
    y = random.randrange(max_y)
    out1[i,:] = src1[x:x+window_size[0],y:y+window_size[1]]
    out2[i,:] = src2[x:x+window_size[0],y:y+window_size[1]]
  return out1, out2

# apply model parameters
def processImg(img, params):
  if 'smpl_type' in params:
    smpl_type    = params['smpl_type']
  else:
    smpl_type    = 'fixed'
  if 'smpl_size' in params:
    val          = int(params['smpl_size'])
    smpl_size    = (val, val)
  else:
    smpl_size    = (32, 32)
  if smpl_type == 'fixed':
    if 'smpl_skip' in params:
      val        = int(params['smpl_skip'])
      smpl_skip  = (val, val)
    else:
      smpl_skip  = smpl_size
    out          = sampleFixed(img, smpl_size, smpl_skip)
  elif smpl_type == 'random':
    if 'smpl_cnt' in param:
      smpl_cnt   = int(parms['smple_cnt'])
    else:
      smpl_cnt   = 1000
    out          = sampleFixed(img, smpl_size, smpl_skip)
  else:
    printf('error: smpl_type needs to be fixed or random')
  return out
  
# get output file params
def getOutParams(params):
  if 'out_dir' in params:
    dst     = params['out_dir']
  else:
    dst     = '.'
  if 'out_format' in params:
    format  = params['out_format']
  else:
    format  = '%s_%05d.%s'
  if 'out_ext' in params:
    ext     = params['out_ext']
  else:
    ext     = 'bmp'
  return dst, format, ext
  
# process files
def processFiles(src, params):
  import ntpath
  import os
  import fileUtils
  import imageUtilsPIL
  files  = fileUtils.getFiles(src)
  dst, format, ext = getOutParams(params)
  for file in files:
    img  = imageUtilsPIL.imageRead(file)
    img  = processImg(img, params)
    head, tail = ntpath.split(file)
    base = os.path.splitext(tail)[0]
    for i in range(np.shape(img)[0]):
      filename   = format % (base, i, ext)
      out  = os.path.join(dst, filename)
      tmp  = img[i,:]
      imageUtilsPIL.imageWrite(out, img[i,:])

# command line interface
if __name__ == "__main__":
  import sys
  import argParse
  params  = argParse.parse(sys.argv[2:])
  processFiles(sys.argv[1], params)
