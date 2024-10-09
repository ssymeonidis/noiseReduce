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
import tensorflow as tf

# upsample fnc
def downsample(x, num_filters, num_conv, dropout):
  for i in range(num_conv):
    x  = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
  x    = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
  if dropout > 0:
    x  = tf.keras.layers.Dropout(dropout)(x)
  return x

# downsample fnc
def upsample(x, num_filters, num_conv, dropout):
  x    = tf.keras.layers.UpSampling2D(size=(2,2))(x)
  if dropout > 0:
    x  = tf.keras.layers.Dropout(dropout)(x)
  for i in range(num_conv):
    x  = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
  return x

# bottleneck fnc
def bottleneck(x, num_filters, num_conv):
  for i in range(num_conv):
    x  = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
  return x

# create tensorflow model
def gen(tensor_size, num_layers, num_filters, num_conv=1, dropout=0, is_final_conv2d=False):
  src  = tf.keras.layers.Input(tensor_size)
  x    = downsample(src, num_filters[0], num_conv, dropout)
  for i in range(num_layers-1):
    x  = downsample(x, num_filters[i+1], num_conv, dropout)
  x    = bottleneck(x, num_filters[num_layers], num_conv)
  for i in range(num_layers-1):
    j  = num_layers - i - 1
    x  = upsample(x, num_filters[j], num_conv, dropout)
  if is_final_conv2d:
    x  = upsample(x, num_filters[0], num_conv, dropout)
  else:
    x  = tf.keras.layers.UpSampling2D(size=(2,2))(x)
  x    = tf.keras.layers.Conv2D(tensor_size[2], kernel_size=(3,3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
  out  = tf.keras.Model(src, x)
  return out

# default configuration
def default(tensor_size):
  return gen(tensor_size, 2, (64, 64, 64, 64))