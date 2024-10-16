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

# expand function
class expand(tf.keras.Layer):
  def call(self, x):
    return tf.expand_dims(x,1)

# enhancement attention modueles
def EAM(src, num_filters):
  x    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(src)
  x    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), dilation_rate=2, padding='same', activation='relu')(x)
  
  y    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), dilation_rate=3, padding='same', activation='relu')(src)
  y    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), dilation_rate=4, padding='same', activation='relu')(y)
  
  z    = tf.keras.layers.Concatenate(axis=-1)([x,y])
  z    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', activation='relu')(z)
  a1   = tf.keras.layers.Add()([z, src])

  z    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', activation='relu')(a1)
  z    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same')(z)
  a2   = tf.keras.layers.Add()([z, a1])
  a2   = tf.keras.layers.Activation('relu')(a2)
  
  z    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', activation='relu')(a2)
  z    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same')(z)
  z    = tf.keras.layers.Conv2D(num_filters, kernel_size=(1,1), padding='same')(z)
  a3   = tf.keras.layers.Add()([z, a2])
  a3   = tf.keras.layers.Activation('relu')(a3)
  
  z    = tf.keras.layers.GlobalAveragePooling2D()(a3)
  z    = expand()(z)
  z    = expand()(z)
  z    = tf.keras.layers.Conv2D(4, kernel_size=(3,3), padding='same', activation='relu')(z)
  z    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', activation='sigmoid')(z)
  z    = tf.keras.layers.Multiply()([z, a3])
  return z

# create tensorflow model
def gen(tensor_size, num_layers, num_filters):
  src  = tf.keras.layers.Input(tensor_size)
  x    = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same')(src)
  for i in range(num_layers):
    x  = EAM(x, num_filters)
  x    = tf.keras.layers.Conv2D(tensor_size[2], kernel_size=(3,3), padding='same')(x)
  x    = tf.keras.layers.Add()([x, src])
  out  = tf.keras.Model(src, x)
  return out

# default configuration
def default(tensor_size):
  return gen(tensor_size, 4, 64)
