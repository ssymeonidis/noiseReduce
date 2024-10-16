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
import json

# save model
def save(filename, model):
  ext = filename.split('.')[-1]
  if ext == 'keras':
    model.save(filename)
  elif ext == 'json':
    config = model.to_json()
    with open(filename, 'w') as json_file:
      json_file.write(config)

# load model
def load(filename):
  ext = filename.split('.')[-1]
  if ext == 'keras':
    return tf.keras.models.load_model(filename)
  elif ext == 'json':
    with open(filename, 'r') as json_file:
      config = json_file.read()
    return tf.keras.models.model_from_json(config)
    
# compile model
def compile(model, learning_rate=1e-03, decay_rate=0.0):
  optimizer  = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
  loss       = tf.keras.losses.MeanSquaredError()
  model.compile(optimizer=optimizer, loss=loss)
      
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

# main function
if __name__ == "__main__":
  import sys
  import datasetUtils
  if len(sys.argv) > 4:
    epochs         = int(sys.argv[4])
  else:
    epochs         = 10
  if len(sys.argv) > 5:
    learning_rate  = float(sys.argv[5])
  else:
    learning_rate  = 1e-03
  model = load(sys.argv[1])
  src_train, out_train, src_test, out_test = datasetUtils.loadSet(sys.argv[2])
  compile(model, learning_rate)
  model.fit(src_train, out_train, epochs=epochs, validation_data=(src_test, out_test))
  save(sys.argv[3])
