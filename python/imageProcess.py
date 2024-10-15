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
import modelUtils
import numpy as np
import math

# main processing function
def processImg(img, model):
  size1     = modelUtils.inputSize(model)
  size2     = np.shape(img)
  blks_y    = math.ceil(float(size2[0]) / float(size1[1]))
  blks_x    = math.ceil(float(size2[1]) / float(size1[2]))
  size3     = (blks_y * size1[1], blks_x * size1[2], size2[2])
  pad_y     = size3[0] - size2[0]
  pad_x     = size3[1] - size2[1]
  img       = np.pad(img,  ((0, pad_y), (0, pad_x), (0,0)), 'symmetric')
  size1     = (blks_y * blks_x, size1[1], size1[2], size1[3])
  tensor    = np.empty(size1, np.float32)
  idx       = 0
  for i in range(blks_y):
    y1      = i  * size1[1]
    y2      = y1 + size1[1]
    for j in range(blks_x):
      x1    = j  * size1[2]
      x2    = x1 + size1[2]
      chip  = img[y1:y2,x1:x2,:]
      tensor[idx,:] = chip
      idx   = idx + 1
  tensor    = tensor / 255
  results   = model.predict(tensor) * 255.0
  results   = results.astype(np.uint8)
  out       = np.empty(size3, np.uint8)
  idx       = 0
  for i in range(blks_y):
    y1      = i  * size1[1]
    y2      = y1 + size1[1]
    for j in range(blks_x):
      x1    = j  * size1[2]
      x2    = x1 + size1[2]
      out[y1:y2,x1:x2,:] = results[idx,:]
      idx   = idx + 1
  return out[:size2[0],:size2[1],:]

# get output file params
def getOutParams(params):
  if 'out_dir' in params:
    dst     = params['out_dir']
  else:
    dst     = '../results'
  if 'out_prefix' in params:
    prefix  = params['out_prefix']
  else:
    prefix  = 'fltr_'
  return dst, prefix

# process files
def processFiles(src, model, params):
  import ntpath
  import os
  import fileUtils
  import imageUtilsPIL
  files  = fileUtils.getFiles(src)
  dst, prefix = getOutParams(params)
  for file in files:
    img  = imageUtilsPIL.imageRead(file)
    img  = processImg(img, model)
    head, tail = ntpath.split(file)
    out  = os.path.join(dst, prefix + tail)
    imageUtilsPIL.imageWrite(out, img)

# command line interface
if __name__ == "__main__":
  import sys
  import argParse
  model   = modelUtils.load(sys.argv[2])
  params  = argParse.parse(sys.argv[3:])
  processFiles(sys.argv[1], model, params)
