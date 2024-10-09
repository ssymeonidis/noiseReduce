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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# read function
def imageRead(filename):
  img  = Image.open(filename)
  data = np.asarray(img)
  img.close()
  return data

# write function
def imageWrite(filename, data):
  img  = Image.fromarray(data)
  img.save(filename)

# display image
def imageDisplay(img):
  plt.figure()
  plt.imshow(img)
  plt.show()

# conver to monochrome
def imageMono(img):
  return np.mean(img, 2)

# command line interface
if __name__ == "__main__":
  import sys
  img = imageRead(sys.argv[1])
  print(img.shape)
  imageDisplay(img)
