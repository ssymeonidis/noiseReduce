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
import glob
import ntpath
import shutil
import os

# get files from directory
def getFiles(path):
  return glob.glob(path)

# copy files to directory w/ prefix
def copyFiles(path, dst, prefix=''):
  files = getFiles(path)
  for file in files:
    head, tail = ntpath.split(file)
    out = os.path.join(dst, prefix + tail)
    shutil.copy2(file, out)

# change permissions
def chmod(path, permission):
  files = getFiles(path)
  for file in files:
    os.chmod(file, permission)

# remove files from directory
def remove(path):
  files = getFiles(path)
  for file in files:
    os.remove(file)
