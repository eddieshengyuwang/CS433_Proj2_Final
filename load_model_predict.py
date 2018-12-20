"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss
Credits: Aurelien Lucchi, ETH Zürich
"""

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
from datetime import datetime
from mask_to_submission import submit_test
import numpy as np

import code

import tensorflow.python.platform

import numpy
import tensorflow as tf

NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TESTING_SIZE = 50
SEED = 42 # Set to None for random seed.
angles = [45, 90,180,270]

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

# load model from most recent

model_dir = "model/" + os.listdir("model")[-1]
print(model_dir)

tf.app.flags.DEFINE_string('train_dir', model_dir,
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches



# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])


# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()


# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:  # bgrd
                l = 0
            else:
                l = 1
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def main(argv=None):  # pylint: disable=unused-argument

    test_data_filename = 'test_set_images/'

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx=0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value*PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image
    def get_prediction(img):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

        return img_prediction

    # Make normal prediction, black white
    def get_prediction_normal(img_prediction):

        w = img_prediction.shape[0]
        h = img_prediction.shape[1]

        nimg = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        nimg8 = img_float_to_uint8(img_prediction)
        nimg[:, :, 0] = nimg8
        nimg[:, :, 1] = nimg8
        nimg[:, :, 2] = nimg8

        return nimg

    def concatenate_images(img, gt_img):
        n_channels = len(gt_img.shape)
        w = gt_img.shape[0]
        h = gt_img.shape[1]
        if n_channels == 3:
            cimg = numpy.concatenate((img, gt_img), axis=1)
        else:
            gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
            gt_img8 = img_float_to_uint8(gt_img)
            gt_img_3c[:, :, 0] = gt_img8
            gt_img_3c[:, :, 1] = gt_img8
            gt_img_3c[:, :, 2] = gt_img8
            img8 = img_float_to_uint8(img)
            cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
        return cimg

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv2 = tf.nn.conv2d(pool,
                             conv2_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

        # Uncomment these lines to check the size of each layer
        # print 'data ' + str(data.get_shape())
        # print 'conv ' + str(conv.get_shape())
        # print 'relu ' + str(relu.get_shape())
        # print 'pool ' + str(pool.get_shape())
        # print 'pool2 ' + str(pool2.get_shape())

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases

        if train:
            summary_id = '_0'
            s_data = get_image_summary(data)
            tf.summary.image('summary_data' + summary_id, s_data)
            s_conv = get_image_summary(conv)
            tf.summary.image('summary_conv' + summary_id, s_conv)
            s_pool = get_image_summary(pool)
            tf.summary.image('summary_pool' + summary_id, s_pool)
            s_conv2 = get_image_summary(conv2)
            tf.summary.image('summary_conv2' + summary_id, s_conv2)
            s_pool2 = get_image_summary(pool2)
            tf.summary.image('summary_pool2' + summary_id, s_pool2)
        return out

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:

        # Restore variables from disk.
        saver.restore(s, FLAGS.train_dir + "/model.ckpt")
        print("Model restored.")

        print("Running prediction on test set")
        now = str(datetime.now()).replace(" ", "_")
        prediction_test_dir = "predictions_test/pred_" + now
        if not os.path.isdir(prediction_test_dir):
            os.mkdir(prediction_test_dir)
            os.mkdir(prediction_test_dir + "/side_by_side")
            os.mkdir(prediction_test_dir + "/overlay")
            os.mkdir(prediction_test_dir + "/normal")
        for i in range(1, TESTING_SIZE + 1):

            imageid = "test_%d" % i
            print(imageid)

            image_filename = test_data_filename + imageid + ".png"
            img = mpimg.imread(image_filename)
            img_prediction = get_prediction(img)

            pimg = concatenate_images(img, img_prediction)
            Image.fromarray(pimg).save(prediction_test_dir + "/side_by_side/ss_" + str(i) + ".png")

            oimg = make_img_overlay(img, img_prediction)
            oimg.save(prediction_test_dir + "/overlay/o_" + str(i) + ".png")

            nimg = get_prediction_normal(img_prediction)
            Image.fromarray(nimg).save(prediction_test_dir + "/normal/pred_test_" + str(i) + ".png")

    # create submission file
    submit_test(prediction_test_dir)

if __name__ == '__main__':
    tf.app.run()