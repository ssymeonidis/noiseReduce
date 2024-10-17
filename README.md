# AI/ML Image Noise Reduction via Tensorflow

This repository implements different Deep Neural Network (DNN) architectures for 
reducing noise in static images.  These DNN architectures are based off articles
published by Sunil Belde, Sharath Solomon, and others.  To generate the training
and validation sets, Gaussian noise was added to public datasets.  However, to
maximize performance, these should be retrained using representative images, i.e.
taken from the target camera in typical scenes/environments.  

The architectures can be extended for processing video by extending the input to
4 dimensions, where the fourth dimension be the several frames, and by modifying 
the Conv2D operations to Conv3D.  The main challenge will be gathering training 
data.


## Neural Network Architectures

The architectures evaluated can be found in the models directory.  'png' files
plot them and 'json' files can be read in Tensorflow using the 'model_from_json' 
function.  The input tensor size is (40,40,3) but genModels.py can be modified
to change this size and to make modifications to the architecture that includes 
changing number of layers, adding dropout layers, etc.


## Training Data

The BSDS500 public dataset, as proposed by Sunil Belde, was used for the initial
development.  The link can be found in the 'Public Dataset' sections below.  The
datasetBSDS500.py script can be used to create this set.  After running, there
should be four 'npy' files created in the 'models' folder.  These contain numpy
arrays that can be read into python using numpy's 'load' function.  This script
will also save intermediary outputs in the 'images' and 'train' folders, where
'images' contain the clean and noise images and 'train' contains the chips with
sizes matched to the Tensorflow model.


## Model Training

The modelUtils.py script can be called via command line to train the models.  An
example using the BSD500 dataset can be found in the genModel.sh shell script.
These will run locally or on the cloud, provided a command line interface.  If
a Python notebook, such as Google Colab, is needed to accelerate training, the
'json' model and 'pny' dataset should be uploaded and the code found in __main__
can be copied/modified into that notebook.  


## Image Processing

The imageProcess.py script can be used to apply the trained model to images.  To
process all the 'src' type the following to create images with the fltr prefix
to be saved in the 'results' folder:  
- python3 imageProcess.py ../model/<model>.keras

## online resources
https://medium.com/analytics-vidhya/noise-removal-in-images-using-deep-learning-models-3972544372d2
https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/

## public datasets
https://github.com/BIDS/BSDS500
