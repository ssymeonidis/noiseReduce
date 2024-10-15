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
import datasetUtils
import imageNoise
import imageSample
import os

# workspace info
src_path_1    = '/home/simeon/Data/BSDS500/BSDS500/data/images/test/*.jpg'
src_path_2    = '/home/simeon/Data/BSDS500/BSDS500/data/images/train/*.jpg'
image_path    = '../images'
image_files   = image_path + '/*.jpg'
train_path    = '../train'
train_files   = train_path + '/*.bmp'
test_samples  = 1000
data_path     = '../models'
data_files    = data_path + '.*.npy'

# config info
noise                 = dict()
noise['out_dir']      = image_path
noise['out_prefix']   = 'src_'
noise['img_level']    = 0 
noise['img_gain']     = 1.0 
noise['img_std']      = 30
sample                = dict()
sample['smpl_type']   = 'fixed'
sample['smpl_size']   = 40
sample['smpl_skip']   = 40
sample['out_dir']     = train_path
sample['out_format']  = '%s_%05d.%s'
sample['out_ext']     = 'bmp'

# clear working directories
fileUtils.remove(image_files)
fileUtils.remove(train_files)
fileUtils.remove(data_files)

# copy repo images to local directory
fileUtils.copyFiles(src_path_1, image_path, 'out_')
fileUtils.copyFiles(src_path_2, image_path, 'out_')

# add noise to images
imageNoise.processFiles(src_path_1, noise)
imageNoise.processFiles(src_path_2, noise)
fileUtils.chmod(image_files, 0o664)

# sample images
imageSample.processFiles(image_files, sample)

# read/split samples
src, out = datasetUtils.readImages(train_path, 'src_', 'out_')
src      = datasetUtils.normalize(src)
out      = datasetUtils.normalize(out)
src_train, src_test = datasetUtils.split(src, test_samples)
out_train, out_test = datasetUtils.split(out, test_samples)
datasetUtils.save(data_path + '/BSDS500_src_train.npy', src_train)
datasetUtils.save(data_path + '/BSDS500_src_test.npy',  src_test)
datasetUtils.save(data_path + '/BSDS500_out_train.npy', out_train)
datasetUtils.save(data_path + '/BSDS500_out_test.npy',  out_test)
