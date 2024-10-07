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

# command line interface
if __name__ == "__main__":
  import imageUtils
  import sys
  img = imageUtils.imageRead(sys.argv[1])
  if len(sys.argv) > 4:
    level = float(sys.argv[3])
    gain  = float(sys.argv[4])
    img   = levelGain(img, level, gain)
  if len(sys.argv) > 2:
    std   = float(sys.argv[2])
  else:
    std   = 50.0
  img = addGaussian(img, std)
  if len(sys.argv) > 5:
    imageUtils.imageWrite(sys.argv[5], img)
  imageUtils.imageDisplay(img)
