import os
from fcn import *
from helper import gen_batch_function

num_classes = 2
images_per_batch = 1
data_dir = './data'
runs_dir = './runs'
train_dir = 'data_road/training2'
viz_dir = './data/data_road/test_set_images'

# set below boolean = True if want to train
train = True
if train:
    input_size = (600, 600)
else:
    input_size = (608, 608)

num_train_examples = 100

get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, train_dir), input_size)
fc_network = FCN(input_size, num_train_examples, viz_dir, images_per_batch, num_classes)
fc_network.optimize(get_batches_fn, num_epochs=10, train=train)
fc_network.inference(runs_dir, data_dir)