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
def layer(x, num_filters):
  x    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same')(x)
  x    = tf.keras.layers.BatchNormalization(axis=-1)(x)
  x    = tf.keras.layers.Activation('relu')(x)
  return x

# create tensorflow model
def gen(tensor_size, num_layers, num_filters):
  src  = tf.keras.layers.Input(tensor_size)
  x    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same')(src)
  x    = tf.keras.layers.Activation('relu')(x)
  for i in range(num_layers):
    x  = layer(x, num_filters)
  x    = tf.keras.layers.Conv2D(tensor_size[2], kernel_size=(3,3), padding='same')(x)
  x    = tf.keras.layers.Subtract()([src, x])
  out  = tf.keras.Model(src, x)
  return out

# default configuration
def default(tensor_size):
  return gen(tensor_size, 16, 64)
