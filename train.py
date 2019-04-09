from imgaug import augmenters as iaa
import pandas as pd
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint
from Unet import Unet
from keras import backend as K
from keras.optimizers import Adam

K.clear_session()

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

def custom_generator(paths, gt_paths, batch_size, num_augs):
    i = 0
    flip_aug = iaa.Fliplr(1)  # flip horizontally
    zoom_aug = iaa.Affine(scale=(1.2, 1.5))  # zoom in 20-50% randomly
    translate_aug = iaa.Affine(translate_percent={"x": (-0.3, 0.3)},
                               mode='symmetric')  # translate l/r/ by 20%

    while True:
        batch_x_aug = []
        batch_y_aug = []
        count = 0
        while count < batch_size / num_augs: # divide by num_augs because data aug multiplies by num_augs
            img_path = paths[i]
            gt_path = gt_paths[i]
            i += 1
            count += 1

            if i == len(paths):
                i = 0

            im = cv2.imread(img_path) / 255.
            # if im.shape != (150,150,3):
            #     print(img_path)
            #     im = cv2.resize(im, (150,150))
            batch_x_aug.append(im)
            im_gt = np.rint(cv2.imread(gt_path, 0) / 255)
            d1, d2 = im_gt.shape
            im_gt = np.reshape(im_gt, (d1, d2, 1))
            batch_y_aug.append(im_gt)

            # im_flip = flip_aug.augment_image(im)
            # batch_x_aug.append(im_flip)
            # batch_y_aug.append(label)

            # im_zoom = zoom_aug.augment_image(im)
            # batch_x_aug.append(im_zoom)
            # batch_y_aug.append(label)

            # im_translate = translate_aug.augment_image(im)
            # batch_x_aug.append(im_translate)
            # batch_y_aug.append(label)

            # cv2.imshow('im', im)
            # cv2.imshow('flip', im_flip)
            # cv2.imshow('zoom', im_zoom)
            # cv2.imshow('translate', im_translate)
            # cv2.waitKey(100)

        batch_x_aug = np.array(batch_x_aug)
        batch_y_aug = np.array(batch_y_aug)

        yield batch_x_aug, batch_y_aug


if __name__ == '__main__':
    train_dir = 'data/training/images/*.png'
    gt_dir = 'data/training/groundtruth/*.png'
    test_dir = 'data/test/*.png'
    train_paths = glob.glob(train_dir)
    gt_paths = glob.glob(gt_dir)
    test_paths = glob.glob(test_dir)

    split_perc = 0.8 # 80% training data
    train_df = pd.DataFrame()
    train_df['path'] = train_paths
    train_df['gt'] = gt_paths
    train_df, val_df = train_test_split(train_df, test_size=1-split_perc)

    batch_size = 8
    save_path = 'model_0.h5'
    num_augs = 1

    gen = custom_generator(train_df['path'].tolist(), train_df['gt'].tolist(), batch_size, num_augs)
    x_val = []
    y_val = []
    for i, row in val_df.iterrows():
        val_im = cv2.imread(row['path']) / 255
        val_gt = np.rint(cv2.imread(row['gt'], 0) / 255)
        d1, d2 = val_gt.shape
        val_gt = np.reshape(val_gt, (d1, d2, 1))
        x_val.append(val_im)
        y_val.append(val_gt)

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    checkpoint = ModelCheckpoint('models/' + save_path, monitor='val_iou', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir='./logs/' + save_path[:-3], write_graph=False)
    callbacks_list = [checkpoint, tensorboard]
    steps_per_epoch = (train_df.shape[0] / batch_size) * num_augs
    input_shape = (400,400,3)
    model = Unet(input_shape)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['acc', f1, iou])
    model.fit_generator(gen, epochs=25, steps_per_epoch= int(steps_per_epoch),
                    callbacks=callbacks_list, validation_data=(x_val, y_val))

