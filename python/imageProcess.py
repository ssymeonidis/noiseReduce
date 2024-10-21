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
import imageSample
import modelUtils
import numpy as np
import math

# calcuate number of blocks
def calcNumBlocks(img_size, win_size, overlap=0):
  img_size  = (img_size[0] - win_size[0], img_size[1] - win_size[1])
  skp_size  = (win_size[0] - overlap, win_size[1] - overlap)
  blks_0    = math.ceil(float(img_size[0]) / float(skp_size[0])) + 1
  blks_1    = math.ceil(float(img_size[1]) / float(skp_size[1])) + 1
  return (blks_0, blks_1)

# calculate size of padded image
def calcPadImageSize(num_blks, win_size, overlap=0):
  skp_size  = (win_size[0] - overlap, win_size[1] - overlap)
  size_0    = win_size[0] + skp_size[0] * (num_blks[0] - 1)
  size_1    = win_size[1] + skp_size[1] * (num_blks[1] - 1)
  return (size_0, size_1)    

# pad image
def padImage(img, num_blks, win_size, overlap=0):
  size      = np.shape(img)
  size_out  = calcPadImageSize(num_blks, win_size, overlap)
  pad_0     = size_out[0] - size[0]
  pad_1     = size_out[1] - size[1]
  img       = np.pad(img, ((0, pad_0), (0, pad_1), (0,0)), 'symmetric')
  return img
  
# generage kernel
def genKernel(win_size, overlap):
  out       = np.ones(win_size)
  if overlap==1:
    out[0,:]   = 0.50
    out[-1,:]  = 0.50
    out[:,0]   = 0.50
    out[:,-1]  = 0.50
    out[0,0]   = 0.25
    out[-1,0]  = 0.25
    out[0,-1]  = 0.25
    out[-1,-1] = 0.25
  if overlap==2:
    out[0,:]   = 0.33
    out[1,:]   = 0.67
    out[-1,:]  = 0.33
    out[-2,:]  = 0.67
    out[:,0]   = 0.33
    out[:,1]   = 0.67
    out[:,-1]  = 0.33
    out[:,-2]  = 0.67
    out[:2,:2]    = 0.25
    out[-2:,:2]   = 0.25
    out[:2,-2:]   = 0.25
    out[-2:,-2:]  = 0.25 
  return out

# reconstruct image (no overlap)
def reconstruct(tensors, num_blks):
  blk_size  = np.shape(tensors)[1:]
  size_out  = calcPadImageSize(num_blks, blk_size[:-1])
  img       = np.empty((size_out[0], size_out[1], blk_size[2]), np.uint8)
  idx       = 0
  for i in range(num_blks[0]):
    y1      = i  * blk_size[0]
    y2      = y1 + blk_size[0]
    for j in range(num_blks[1]):
      x1    = j  * blk_size[1]
      x2    = x1 + blk_size[1]
      img[y1:y2,x1:x2,:] = tensors[idx,:]
      idx   = idx + 1
  return img

# reconstruct image w/ overlap
def reconstructOverlap(tensors, num_blks, overlap):
  blk_size  = np.shape(tensors)[1:]
  size_out  = calcPadImageSize(num_blks, blk_size[:-1])
  img       = np.zeros((size_out[0], size_out[1], blk_size[2]), np.uint8)
  kernel    = genKernel(blk_size, overlap)
  skp_size  = (blk_size[0]-overlap, blk_size[1]-overlap)
  idx       = 0
  for i in range(num_blks[0]):
    y1      = i  * skp_size[0]
    y2      = y1 + blk_size[0]
    for j in range(num_blks[1]):
      x1    = j  * skp_size[1]
      x2    = x1 + blk_size[1]
      img[y1:y2,x1:x2,:] = img[y1:y2,x1:x2,:] + kernel * tensors[idx,:]
      idx   = idx + 1
  return img

# main processing function
def processImgGrid(img, model, overlap=0):
  win_size  = modelUtils.inputSize(model)[1:3]
  img_size  = np.shape(img)
  num_blks  = calcNumBlocks(img_size, win_size, overlap)
  img       = padImage(img, num_blks, win_size, overlap)
  skp_size  = (win_size[0] - overlap, win_size[1] - overlap)
  tensors   = imageSample.sampleFixed(img, win_size, skp_size)
  tensors   = tensors.astype(np.float32) / 255
  results   = modelUtils.run(model, tensors)
  results   = 255.0 * np.clip(results, 0.0, 1.0)
  results   = results.astype(np.uint8)
  if overlap==0:
    out     = reconstruct(results, num_blks)
  else:
    out     = reconstructOverlap(results, num_blks, overlap)
  return out[:img_size[0],:img_size[1],:]

# get output file params
def getParams(params):
  if 'overlap' in params:
    overlap = int(params['overlap'])
  else:
    overlap = 0
  if 'out_dir' in params:
    dst     = params['out_dir']
  else:
    dst     = '../results'
  if 'out_prefix' in params:
    prefix  = params['out_prefix']
  else:
    prefix  = 'fltr_'
  return overlap, dst, prefix

# process files
def processFiles(src, model, params):
  import ntpath
  import os
  import fileUtils
  import imageUtilsPIL
  files  = fileUtils.getFiles(src)
  overlap, dst, prefix = getParams(params)
  for file in files:
    img  = imageUtilsPIL.imageRead(file)
    img  = processImgGrid(img, model, overlap)
    head, tail = ntpath.split(file)
    out  = os.path.join(dst, prefix + tail)
    imageUtilsPIL.imageWrite(out, img)
    print('finished' + out + '...')

# command line interface
if __name__ == "__main__":
  import sys
  import argParse
  model   = modelUtils.load(sys.argv[1])
  params  = argParse.parse(sys.argv[2:])
  if 'src' in params:
    src     = params['src']
  else:
    src     = '../images/src*.jpg'
  processFiles(src, model, params)
