# Fully Convolutional Networks for Image Road Segmentation
##### Rebecca Cheng, Eddie Wang, Sandra Yang

This project aims to perform semantic image segmentation from Google satellite images for roads. We used a fully convolutional network (FCN) architecture with binary classification to predict 16x16 patches of satellite images as either road (1) or non-road (0).

### Prerequisites

To install all the libraries required for the repository to run, execute the following Linux command:

```
pip install tensorflow, numpy, scipy, tqdm, opencv-python
```

Afterwards, you will need to download the VGG-16 from [this link](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz): and save it to the `pretrained_weights` folder. 

Next, you need to download our most recently trained weights from [this link](https://drive.google.com/file/d/1iXWgUc5SZx3sZufvGgNFw7Wgj3bBALkN/view?usp=sharing) and place it in the `checkpoints/kitti/900/` directory. There should be four files that are extracted from the link, make sure all of them are extracted to the directory!

### Running

We ran our training on Google Cloud Platform NVIDIA Tesla K80 GPU, which took around 2 hours to train. If you want to train the model from scratch, make sure to set `train = True` on line 13 in the `run.py` file:

```
...
viz_dir = './data/data_road/test_set_images'

# set below boolean = True if want to train
train = True
...
```

Alternatively, you can leave the `run.py` file as is, which will use our pre-trained FCN model to predict outputs for the test images. The submission file will be found in the `runs/<highest number>/` directory called `fcn_16_patch8.csv`. 

## Description of files
### FCN Model
- `fcn.py`: this file takes the downloaded pre-trained VGG-16 CNN and converts it into an FCN. It also contains the training and prediction code. 
- `helper.py`: this file contains many useful helper functions to preprocess images and save checkpoints during training.
- `run.py`: wrapper file that trains and predicts road segmentation images using the FCN model. It will either train from scratch or use our most recently trained weights to predict test images based on user setting
- `loss.py` and `model_utils.py`: helper files for `fcn.py` 

### CNN Model
We also trained a CNN as a baseline model to compare with the FCN in the report. Below are files used to run and train the CNN model. 
-  `tf_aerial_images.py`: run this first to train the CNN model
- `load_model_predict`: run this after to predict test images