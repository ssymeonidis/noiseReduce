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

list_dir  = "../train"
prefix1   = "src"
prefix2   = "out"
sformat   = "%05d.bmp"
format1   = list_dir + "/" + prefix1 + sformat
format2   = list_dir + "/" + prefix2 + sformat

# get dimensions of output array
def getDim(src, num_samples):
  size  = np.shape(src)
  dimIn = src.ndim
  if dimIn == 2:
    dim = (num_samples, size[0], size[1])
  else:
    dim = (num_samples, size[0], size[1], size[2])
  return dim

# read training iamges
def readImages(folder, src_prefix, out_prefix):
  import ntpath
  import os
  import fileUtils
  import imageUtilsPIL
  files_src     = fileUtils.getFiles(folder + '/' + src_prefix + '*')
  img           = imageUtilsPIL.imageRead(files_src[0])
  size          = getDim(img, len(files_src))
  src           = np.empty(size, img.dtype)
  out           = np.empty(size, img.dtype)
  idx           = 0
  for file_src in files_src:
    head, tail  = ntpath.split(file_src)
    file_out    = os.path.join(folder, out_prefix + tail[len(src_prefix):])
    src[idx,:]  = imageUtilsPIL.imageRead(file_src)
    out[idx,:]  = imageUtilsPIL.imageRead(file_out)
    idx         = idx + 1
  return src, out

# normalize dataset
def normalize(data):
  return data.astype(np.float32) / 255.0

# split dataset into training and test subsets
def split(dataset, num_test_samples):
  mid           = np.shape(dataset)[0] - num_test_samples
  train         = dataset[:mid,:]
  test          = dataset[mid:,:]
  return train, test

# save numpy array to file (recommend npy ext)
def save(filename, data):
  np.save(filename, data, allow_pickle=False)

# load numpy array from file (recommend npy ext)
def load(filename):
  return np.load(filename, allow_pickle=False)
