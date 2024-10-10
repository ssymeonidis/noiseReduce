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


# import library
import fileUtils

# workspace info
src_path_1 = '/home/simeon/Data/BSDS500/BSDS500/data/images/test/*.jpg'
src_path_2 = '/home/simeon/Data/BSDS500/BSDS500/data/images/train/*.jpg'
image_path = '../images'

# copy repo images to local directory
fileUtils.copyFiles(src_path_1, image_path, 'out_')
fileUtils.copyFiles(src_path_2, image_path, 'out_')
