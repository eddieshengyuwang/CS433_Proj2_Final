# U-Net for Image Road Segmentation
##### Rebecca Cheng, Eddie Wang, Sandra Yang

This project aims to perform semantic image segmentation from Google satellite images for roads. We first used a fully convolutional network (FCN) architecture with binary classification to predict 16x16 patches of satellite images as either road (1) or non-road (0), as found in [this report](https://github.com/eddieshengyuwang/CS433_Proj2_Final/blob/master/report_FCN_OLD.pdf). We later found that a U-Net architecture performed much better with 0.884 F1-Score compared to 0.78 with FCN (unfortunately no written report on the U-Net model). Note that this repository only implements the project using the U-Net approach.

Example of original training image, ground truth, model prediction:
![](https://github.com/eddieshengyuwang/CS433_Proj2_Final/blob/master/data/pred_example.png)

### Prerequisites

The project requires Python 3 and the libraries found in `requirements.txt`:

```
pip install -r requirements.txt
```

### Training

Training code is found in the `train.py` file. It requires a mandatory `--save_path` argument specifying the model save name in the *models/* directory and an optional `--load_path` argument specifying the model load name in the *models/* directory if you want to resume training from a saved point. If `--load_path` is not specified, then the model will train from scratch (which took around 40 min for 15 epochs on a Google Cloud Platform NVIDIA Tesla K80 GPU). Below are example calls:

```
# train from scratch
python train.py --save_path model_1_5.h5

# train from saved model
python train.py --load_path model_1_4.h5 --save_path model_1_5.h5
```

The training script uses a custom generator (`custom_generator`) to continually generate augmented images. For our training the augmentations are: 1 original image, 5 rotations in [60, 120, 180, 240, 300] degrees, 1 horizontal flip, and 1 vertical flip (8 augmentations in total). If you want to modify image augmentations, be sure to change the `num_augs` argument in the `custom_generator` function accordingly. 

The training script also separates the training set data into 80/20 split between training and validation sets. Best model predictions on validation set per epoch are based on *intersection over union (Jaccard index)* are saved in the *models/<save_path>* directory. Lastly, training progress is tracked through Tensorboard callbacks. To view the graphs, run the following command on Terminal:  

```
tensorboard --logdir logs/
```

And navigate on a web browser to the link Tensorboard indicated on Terminal (example for validation IOU):

![](https://github.com/eddieshengyuwang/CS433_Proj2_Final/blob/master/data/satImage_078.png)

### Predictions
To make predictions on the test set, use `predict.py`.  It requires a mandatory argument `model_name` indicating which model in the *models/*  directory and optional arguments `--use_training` and `--debug` if you want to predict using the training set and want to see the prediction results, respectively. Note that if either is set, then the script will **not** generate submission csvs. Otherwise a submission csv with the current timestamp will be created in the *submissions/*  directory. 

```
# predict on training set and view them 
python predict.py --model_name model_1_5.h5 --debug --use_training

# generate submission csv
python predict.py --model_name model_1_5.h5
```

### Additional files
- `Unet.py`: U-Net model structure code.
- `ml_utils.py`: Contains useful helper functions like calculating IOU, etc.
- `mask_to_submission.py`: Code to convert prediction images to a submission csv file.
