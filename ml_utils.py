from keras import backend as K
import cv2
import numpy as np
from imgaug import augmenters as iaa
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # add parsing arguments below
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--model_name', type=str, default='',
                        help='Model name.')
    parser.add_argument('--save_path', type=str, default='',
                        help='Save path.')
    parser.add_argument('--load_path', type=str, default='',
                        help='Load path.')
    return parser.parse_args()

def iou(y_true, y_pred):
    intersection = y_true * y_pred
    not_true = 1 - y_true
    union = y_true + not_true * y_pred
    return K.sum(intersection) / K.sum(union)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def custom_generator(paths, labels, batch_size, num_augs, debug=False):
    i = 0

    # _____________________________add image_augmenters below_______________________________________
    # flip_aug = iaa.Fliplr(1)  # flip horizontally
    # zoom_aug = iaa.Affine(scale=(1.2, 1.5))  # zoom in 20-50% randomly
    # translate_aug = iaa.Affine(translate_percent={"x": (-0.3, 0.3)},
    #                            mode='symmetric')  # translate l/r/ by 20%

    while True:
        batch_x_aug = []
        batch_y_aug = []
        count = 0
        while count < batch_size / num_augs: # divide by num_augs because data aug multiplies by num_augs
            img_path = paths[i]
            i += 1
            count += 1

            if i == len(paths):
                i = 0

            im = cv2.imread(img_path) / 255.
            batch_x_aug.append(im)
            d1, d2 = im_gt.shape
            im_gt = np.reshape(im_gt, (d1, d2, 1))
            batch_y_aug.append(labels[i])

            # __________________________apply image augmentations below________________________________
            # im_flip = flip_aug.augment_image(im)
            # batch_x_aug.append(im_flip)
            # batch_y_aug.append(label)

            if debug:
                cv2.imshow('im', im)
                #cv2.imshow('flip', im_flip)
                cv2.waitKey(100)

        batch_x_aug = np.array(batch_x_aug)
        batch_y_aug = np.array(batch_y_aug)

        yield batch_x_aug, batch_y_aug

def train_val_split(og_df, train_percentage):
    labels = og_df.action_after.unique()
    train_df = pd.DataFrame(columns=og_df.columns)
    val_df = pd.DataFrame(columns=og_df.columns)

    for label in labels:
        df_label = og_df[og_df.action_after == label]
        np.random.seed(42)
        rows_train = np.random.rand(len(df_label)) < train_percentage
        train_df = train_df.append(df_label[rows_train])
        val_df = val_df.append(df_label[~rows_train])

    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    return train_df, val_df
