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
import datasetUtils
import modelUtils
import modelAutoEncoder
import tensorflow as tf
import numpy as np

# model parameters
learning_rate  = 1e-03
epochs         = 10

# read training/test data
src_train      = datasetUtils.load('../models/BSDS500_src_train.npy')
src_test       = datasetUtils.load('../models/BSDS500_src_test.npy')
out_train      = datasetUtils.load('../models/BSDS500_out_train.npy')
out_test       = datasetUtils.load('../models/BSDS500_out_test.npy')
tensor_size    = np.shape(src_train)[1:]

# read and compile auto-enocoder model
model          = modelAutoEncoder.gen(tensor_size, 2, (64, 64, 64, 64))
optimizer      = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
loss           = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optimizer, loss=loss)
history        = model.fit(src_train, out_train, epochs=epochs, validation_data=(src_test, out_test))
# test_loss, test_acc = model.evaluate(src_test, out_test, verbose=2)
modelUtils.save('../models/BSDS500_autoencoder.keras', model)
