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
import modelMLCNN
import modelAutoEncoder
import modelUnet
import modelDNCNN
import modelRIDNET
import modelUtils
import datasetUtils
import os
import sys
import numpy as np

# extract tensorflow size
tmp = sys.argv[1].split(',')
if len(tmp)==3:
  size = (int(tmp[0]), int(tmp[1]), int(tmp[2])) 
elif os.path.exists(sys.argv[1]):
  tmp  = datasetUtils.load(sys.argv[1])
  size = np.shape(tmp)[1:]
else:
  tmp1, tmp2, tmp3, tmp4 = datasetUtils.loadSet(sys.argv[1])
  size = np.shape(tmp1)[1:]
print(size)

# create models
model1 = modelMLCNN.gen(size, 6, 64)
modelUtils.save('../models/arch_MLCNN.json', model1)
modelUtils.plotModel(model1, '../models/arch_MLCNN.png')
model2 = modelAutoEncoder.gen(size, 2, 64)
modelUtils.save('../models/arch_AutoEncoder.json', model2)
modelUtils.plotModel(model2, '../models/arch_AutoEncoder.png')
model3 = modelUnet.gen(size, 2, (32, 64, 128))
modelUtils.save('../models/arch_Unet.json', model3)
modelUtils.plotModel(model3, '../models/arch_Unet.png')
model4 = modelDNCNN.gen(size, 8, 64)
modelUtils.save('../models/arch_DNCNN.json', model4)
modelUtils.plotModel(model4, '../models/arch_DNCNN.png')
model5 = modelRIDNET.gen(size, 4, 64)
modelUtils.save('../models/arch_RIDNET.json', model5)
modelUtils.plotModel(model5, '../models/arch_RIDNET.png')
