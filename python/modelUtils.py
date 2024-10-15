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

# save model
def save(filename, model):
  model.save(filename)

# load model
def load(filename):
  return tf.keras.models.load_model(filename)

# get input tensor size
def inputSize(model):
  config = model.get_config()
  return config["layers"][0]["config"]["batch_shape"]

# print layer summary
def printSummary(model):
  model.summary()

# generates layer image
def plotModel(model, filename='../model/model.png'):
  tf.keras.utils.plot_model(model, to_file=filename, show_shapes=True)
