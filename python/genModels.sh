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


# generate deataset
python3 datasetBSDS500.py

# train tensorflow models
python3 modelUtils.py ../models/arch_MLCNN.json        data=../models/BSDS500_  keras=../models/model_MLCNN_BSDS500.keras        >> ../models/model_MLCNN_BSDS500.log
python3 modelUtils.py ../models/arch_AutoEncoder.json  data=../models/BSDS500_  keras=../models/model_AutoEncoder_BSDS500.keras  >> ../models/model_AutoEncoder_BSDS500.log
python3 modelUtils.py ../models/arch_Unet.json         dsta=../models/BSDS500_  keras=../models/model_Unet_BSDS500.keras         >> ../models/model_Unet_BSDS500.log
python3 modelUtils.py ../models/arch_DNCNN.json        data=../models/BSDS500_  keras=../models/model_DNCNN_BSDS500.keras        >> ../models/model_DNCNN_BSDS500.log
python3 modelUtils.py ../models/arch_RIDNET.json       data=../models/BSDS500_  keras=../models/model_RIDNET_BSDS500.keras       >> ../models/model_RIDNET_BSDS500.log

# covert to tensorflow lite
python3 modelUtils.py ../models/model_MLCNN_BSDS500.keras        tflite=../models/model_MLCNN_BSDS500.tflite        type=float16
python3 modelUtils.py ../models/model_AutoEncoder_BSDS500.keras  tflite=../models/model_AutoEncoder_BSDS500.tflite  type=float16
python3 modelUtils.py ../models/model_Unet_BSDS500.keras         tflite=../models/model_Unet_BSDS500.tflite         type=float16
python3 modelUtils.py ../models/model_DNCNN_BSDS500.keras        tflite=../models/model_DNCNN_BSDS500.tflite        type=float16
python3 modelUtils.py ../models/model_RIDNET_BSDS500.keras       tflite=../models/model_RIDNET_BSDS500.tflite       type=float16
