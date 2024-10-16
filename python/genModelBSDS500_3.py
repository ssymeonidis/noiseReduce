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
import modelDNCNN
import tensorflow as tf
import numpy as np

# model parameters
learning_rate  = 1e-03
epochs         = 10

# read training/test data
src_train, out_train, src_test, out_test = datasetUtils.loadSet('../models/BSDS500_')
tensor_size    = np.shape(src_train)[1:]

# read and compile auto-enocoder model
model          = modelDNCNN.gen(tensor_size, 10, 64)
modelUtils.compile(model, learning_rate)
history        = model.fit(src_train, out_train, epochs=epochs, validation_data=(src_test, out_test))
modelUtils.save('../models/BSDS500_modelDNCNN.keras', model)
