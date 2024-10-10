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
import imageNoise
import os

# workspace info
src_path_1  = '/home/simeon/Data/BSDS500/BSDS500/data/images/test/*.jpg'
src_path_2  = '/home/simeon/Data/BSDS500/BSDS500/data/images/train/*.jpg'
image_path  = '../images'
noise                = dict()
noise['out_dir']     = image_path
noise['out_prefix']  = 'src_'
noise['img_level']   = 32
noise['img_gain']    = 0.75
noise['img_std']     = 30

# copy repo images to local directory
fileUtils.copyFiles(src_path_1, image_path, 'out_')
fileUtils.copyFiles(src_path_2, image_path, 'out_')

# add noise to images
imageNoise.processFiles(src_path_1, noise)
imageNoise.processFiles(src_path_2, noise)
fileUtils.chmod(image_path, 0o664)
