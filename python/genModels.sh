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


python3 datasetBSDS500.py
python3 modelUtils.py ../models/arch_MLCNN.json         ../models/BSDS500_  ../models/model_MLCNN_BSDS500.keras
python3 modelUtils.py ../models/arch_AutoEncoder.json   ../models/BSDS500_  ../models/model_AutoEncoder_BSDS500.keras
python3 modelUtils.py ../models/arch_Unet.json          ../models/BSDS500_  ../models/model_Unet_BSDS500.keras
python3 modelUtils.py ../models/arch_DNCNN.json         ../models/BSDS500_  ../models/model_DNCNN_BSDS500.keras
