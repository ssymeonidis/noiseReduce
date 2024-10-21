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
import numpy as np
import tensorflow as tf
import json
import argParse

# get model type 
def getType(model):
  type_str = str(type(model)).split('.')
  if 'functional' in type_str:
    return 'keras'
  elif 'lite' in type_str:
    return 'tflite'
  else:
    print('error: model not supported')

# save model
def save(filename, model):
  ext = filename.split('.')[-1]
  if ext == 'json':
    config = model.to_json()
    with open(filename, 'w') as json_file:
      json_file.write(config)
  elif ext == 'keras':
    model.save(filename)
  elif ext == 'tflite':
    open(filename, 'wb').write(model)
  else:
    printf('error: unknown extension')

# load model
def load(filename):
  ext = filename.split('.')[-1]
  if ext == 'json':
    with open(filename, 'r') as json_file:
      config = json_file.read()
    return tf.keras.models.model_from_json(config)
  elif ext == 'keras':
    return tf.keras.models.load_model(filename)
  elif ext == 'tflite':
    return tf.lite.Interpreter(model_path=filename)
  else:
    printf('error: unknown extension')
    
# get input tensor size
def inputSize(model):
  model_type = getType(model)
  if model_type == 'keras':
    config = model.get_config()
    return config["layers"][0]["config"]["batch_shape"]
  elif model_type == 'tflite':
    return model.get_input_details()[0]['shape']
  else:
    print('error: model not supported')

# get outut tensor size
def outputSize(model):
  model_type = getType(model)
  if model_type == 'tflite':
    return model.get_output_details()[0]['shape']
  else:
    print('error: model not supported')
   
# compile model
def compile(model, learning_rate=1e-03, decay_rate=0.0):
  optimizer  = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
  loss       = tf.keras.losses.MeanSquaredError()
  model.compile(optimizer=optimizer, loss=loss)
  
# train model
def train(model, epochs, src_train, out_train, src_test=None, out_test=None):
  if np.size(src_test) > 1 and np.size(out_test) > 1:
    model.fit(src_train, out_train, epochs=epochs, validation_data=(src_test, out_test))
  else:
    model.fit(src_train, out_train, epochs=epochs)

# convert model
def convert(model, dtype=''):
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
  ]
  # if dtype == 'float16':
  #   converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
  return converter.convert()

# run model
def run(model, tensors):
  model_type = getType(model)
  if model_type == 'keras':
    return model.predict(tensors)
  elif model_type == 'tflite':
    size_in         = np.shape(tensors)
    size_out        = outputSize(model)
    size_out        = (size_in[0], size_out[1], size_out[2], size_out[3])
    input_details   = model.get_input_details()
    output_details  = model.get_output_details()
    model.resize_tensor_input(input_details[0]['index'], size_in)
    model.resize_tensor_input(output_details[0]['index'], size_out)
    model.allocate_tensors()
    model.set_tensor(input_details[0]['index'], tensors)
    model.invoke()
    return model.get_tensor(output_details[0]['index'])
  else:
    print('error: model not supported')
      
# print layer summary
def printSummary(model):
  model.summary()

# generates layer image
def plotModel(model, filename='../model/model.png'):
  tf.keras.utils.plot_model(model, to_file=filename, show_shapes=True)

# get output file params
def getParams(params):
  if 'keras' in params:
    keras     = params['keras']
  else:
    keras     = ''
  if 'tflite' in params:
    tflite    = params['tflite']
  else:
    tflite    = ''
  if 'data' in params:
    data      = params['data']
  else:
    data      = ''
  if 'epochs' in params:
    epochs    = int(params['epochs'])
  else:
    epochs    = 10
  if 'rate' in params:
    rate      = float(params['rate'])
  else:
    rate      = 1e-03
  if 'dtype' in params:
    dtype     = params['dtype']
  else:
    dtype     = ''
  return keras, tflite, data, epochs, rate, dtype

# main function
if __name__ == '__main__':
  import sys
  import datasetUtils
  model   = load(sys.argv[1])
  params  = argParse.parse(sys.argv[2:])
  keras, tflite, data, epochs, rate, dtype = getParams(params)
  if len(data) > 0:
    src_train, out_train, src_test, out_test = datasetUtils.loadSet(data)
    compile(model, rate)
    train(model, epochs, src_train, out_train, src_test, out_test)
  if len(keras) > 0:
    save(keras, model)
  if len(tflite) > 0:
    model_lite = convert(model, dtype)
    save(tflite, model_lite)
