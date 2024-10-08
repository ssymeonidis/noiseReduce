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
import imageUtils
import imageNoise
import imageSample

format1   = "../train/src%05d.bmp"
format2   = "../train/out%05d.bmp"

# process single image
def processImage(filename, num_samples, window_size, params, idx):
  src2    = imageUtils.imageRead(filename)
  if 'is_mono' in params and params['is_mono']:
    src2  = imageUtils.imageMono(src2)
  src1    = imageNoise.applyParams(src2, params)
  (data1, data2) = imageSample.sample2(src1, src2, window_size, num_samples)
  for i in range(num_samples):
    out1  = format1 % idx
    out2  = format2 % idx
    imageUtils.imageWrite(out1, data1[i,:])
    imageUtils.imageWrite(out2, data2[i,:])
    idx   = idx + 1
    
# process multiple images
def processList(filename_list, num_samples_list, window_size, params):
  idx = 0
  for i in range(len(filename_list)):
    processImage(filename_list[i], num_samples_list[i], window_size, params, idx)
    idx = idx + num_samples_list[i]

# parse command line args
def parse(args):
  filename_list     = list()
  num_samples_list  = list()
  idx               = 0
  for arg in args:
    is_param = arg.rfind('=')
    if is_param < 0 and idx % 2 == 0:
      filename_list.append(arg)
    elif is_param < 0:
      num_samples_list.append(int(arg))
    else:
      break
    idx += 1
  params = argParse.parse(args[idx:])     
  return (filename_list, num_samples_list, params)

# command line interface
if __name__ == "__main__":
  import sys
  import argParse
  (filename_list, num_samples_list, params) = parse(sys.argv[1:])
  window_size = (int(params['window_size']), int(params['window_size']))
  processList(filename_list, num_samples_list, window_size, params)
