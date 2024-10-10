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

# gaussian noise adder
def addGaussian(img, std):
  size  = img.shape
  noise = np.random.normal(0, std, size)
  out   = img.astype(float) + noise
  out   = np.clip(out, 0, 255)
  return out.astype(img.dtype)

# apply level and gain
def levelGain(img, level, gain):
  out   = img.astype(float)
  out   = gain * out + level;
  return out.astype(img.dtype)

# apply model parameters
def processImg(img, params):
  if 'img_level' in params and 'img_gain' in params:
    level = float(params['img_level'])
    gain  = float(params['img_gain'])
    img   = levelGain(img, level, gain)
  if 'img_std' in params:
    std   = float(params['img_std'])
    img   = addGaussian(img, std)
  return img
  
# get output file params
def getOutParams(params):
  if 'out_dir' in params:
    dst     = params['out_dir']
  else:
    dst     = '.'
  if 'out_prefix' in params:
    prefix  = params['out_prefix']
  else:
    prefix  = ''
  return dst, prefix
  
# process files
def processFiles(src, params):
  import ntpath
  import os
  import fileUtils
  import imageUtilsPIL
  files  = fileUtils.getFiles(src)
  dst, prefix  = getOutParams(params)
  for file in files:
    img  = imageUtilsPIL.imageRead(file)
    img  = processImg(img, params)
    if 'out_disp' in params:
      imageUtilsPIL.imageDisplay(img)
      break
    else:  
      head, tail = ntpath.split(file)
      out = os.path.join(dst, prefix + tail)
      imageUtilsPIL.imageWrite(out, img)

# command line interface
if __name__ == "__main__":
  import sys
  import argParse
  params  = argParse.parse(sys.argv[2:])
  processFiles(sys.argv[1], params)
