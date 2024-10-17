# AI/ML Based Image Noise Reduction using Tensorflow

This repository implements different Deep Neural Network (DNN) architectures for 
reducing noise in static images.  These DNN architectures were based of articles
published by Sunil Belde, Sharath Solomon, and others.  To generate the training
and validation sets, Gaussian noise was added to public datasets.  However, to
maximize performance, these should be retrained using representative images.  

The architectures can be extended for processing video by extending the input to
a 4-dimensional input, where the fourth dimension be the several frames, and by
modifying the Conv2D operations to Conv3D.  The main challenge will be gathering
training data.


## Neural Network Architectures

The architectures evaluated can be found in the models directory.  'png' files
plot all the architecture and 'json' files can be read into Tensorflow using the 
'model_from_json' function.  The input tensor size is (40,40,3) but genModels.py
can be used to change this size and make modifications to the architecture that
includes changing number of layers, adding dropout layers, etc.


## online resources
https://medium.com/analytics-vidhya/noise-removal-in-images-using-deep-learning-models-3972544372d2
https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/

## public datasets
https://github.com/BIDS/BSDS500
