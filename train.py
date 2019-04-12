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
from ml_utils import f1, iou
from keras.models import load_model

K.clear_session()

def custom_generator(paths, gt_paths, batch_size, num_augs, target_shape, debug=False):
    # num_augs = 8; 1 original + 5 rotations + 2 flips, see below data augmentations
    i = 0
    fliplr_aug = iaa.Fliplr(1)  # flip horizontally
    flipud_aug = iaa.Flipud(1) # flip vertically

    assert batch_size >= num_augs, "batch size has to be greater/eq than num augs you're gonna do!"
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
            im_gt = cv2.imread(gt_path, 0)

            # reflect border to make image 620x620
            pad_ud = (target_shape[0] - im.shape[0]) // 2
            pad_lr = (target_shape[1] - im.shape[1]) // 2
            im = cv2.copyMakeBorder(im, pad_ud, pad_ud, pad_lr, pad_lr, cv2.BORDER_REFLECT)
            im_gt = cv2.copyMakeBorder(im_gt, pad_ud, pad_ud, pad_lr, pad_lr, cv2.BORDER_REFLECT)

            im = cv2.resize(im, (300, 300))
            im_gt = cv2.resize(im_gt, (300, 300))
            im_gt = np.rint(im_gt / 255)

            assert np.unique(im_gt).shape[0] == 2

            batch_x_aug.append(im)
            batch_y_aug.append(im_gt)

            if debug:
                cv2.imshow('c', im)
                cv2.waitKey(100)
                cv2.imshow('gt', im_gt)
                cv2.waitKey(100)


            # rotate
            rows, cols = im.shape[0], im.shape[1]
            degrees = [60, 120, 180, 240, 300] # generate five more images
            for degree in degrees:
                M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degree, 1)
                rotate_img = cv2.warpAffine(im, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
                rotate_gt = cv2.warpAffine(im_gt, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
                rotate_gt = np.rint(rotate_gt)
                assert np.unique(rotate_gt).shape[0] == 2

                batch_x_aug.append(rotate_img)
                batch_y_aug.append(rotate_gt)

                if debug:
                    cv2.imshow('c2 {}'.format(degree), rotate_img)
                    cv2.waitKey(100)
                    cv2.imshow('gt2 {}'.format(degree), rotate_gt)
                    cv2.waitKey(100)

            # flip image lr, ud
            im_fliplr = fliplr_aug.augment_image(im)
            batch_x_aug.append(im_fliplr)
            gt_fliplr = fliplr_aug.augment_image(im_gt)
            batch_y_aug.append(gt_fliplr)
            assert np.unique(gt_fliplr).shape[0] == 2

            im_flipud = flipud_aug.augment_image(im)
            batch_x_aug.append(im_flipud)
            gt_flipud = flipud_aug.augment_image(im_gt)
            batch_y_aug.append(gt_flipud)
            assert np.unique(gt_flipud).shape[0] == 2

            if debug:
                cv2.imshow('lr', im_fliplr)
                cv2.waitKey(100)
                cv2.imshow('lr_gt', gt_fliplr)
                cv2.waitKey(100)

                cv2.imshow('ud', im_flipud)
                cv2.waitKey(100)
                cv2.imshow('ud_gt', gt_flipud)
                cv2.waitKey(100)

        batch_x_aug = np.array(batch_x_aug)
        batch_y_aug = np.array(batch_y_aug)
        batch_y_aug = np.reshape(batch_y_aug, (*batch_y_aug.shape, 1))

        yield batch_x_aug, batch_y_aug


if __name__ == '__main__':
    train_dir = 'data/training/images/*.png'
    gt_dir = 'data/training/groundtruth/*.png'
    test_dir = 'data/test/*.png'
    train_paths = glob.glob(train_dir)
    gt_paths = glob.glob(gt_dir)
    test_paths = glob.glob(test_dir)

    split_perc = 0.7 # 70% training data
    train_df = pd.DataFrame()
    train_df['path'] = train_paths
    train_df['gt'] = gt_paths
    train_df, val_df = train_test_split(train_df, test_size=1-split_perc)

    batch_size = 8
    save_path = 'model_1_2.h5'
    target_shape = (608, 608)
    debug = False
    num_augs = 8 # this number is determined manually by the # of data augs you do in custom_generator

    gen = custom_generator(train_df['path'].tolist(), train_df['gt'].tolist(),
                           batch_size, num_augs, target_shape, debug)
    x_val = []
    y_val = []
    for i, row in val_df.iterrows():
        val_im = cv2.imread(row['path']) / 255
        val_gt = cv2.imread(row['gt'], 0)

        # reshape image to be 608x608
        pad_ud = (target_shape[0] - val_im.shape[0]) // 2
        pad_lr = (target_shape[1] - val_im.shape[1]) // 2
        val_im = cv2.copyMakeBorder(val_im, pad_ud, pad_ud, pad_lr, pad_lr, cv2.BORDER_REFLECT)
        val_gt = cv2.copyMakeBorder(val_gt, pad_ud, pad_ud, pad_lr, pad_lr, cv2.BORDER_REFLECT)

        # reshape to 300x300
        val_im = cv2.resize(val_im, (300, 300))
        val_gt = cv2.resize(val_gt, (300, 300))
        val_gt = np.rint(val_gt / 255)
        d1, d2 = val_gt.shape
        val_gt = np.reshape(val_gt, (d1, d2, 1))
        assert np.unique(val_gt).shape[0] == 2
        x_val.append(val_im)
        y_val.append(val_gt)

# ------------------------------------
#         cv2.imshow('c', val_im)
#         cv2.waitKey(100)
#         cv2.imshow('gt', val_gt)
#         cv2.waitKey(100)
#
#         fliplr_aug = iaa.Fliplr(1)  # flip horizontally
#         flipud_aug = iaa.Flipud(1)  # flip vertically
#
#         # rotate
#         rows, cols = val_im.shape[0], val_im.shape[1]
#         degrees = [60, 120, 180, 240, 300]  # generate five more images
#         for degree in degrees:
#             M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degree, 1)
#             rotate_img = cv2.warpAffine(val_im, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
#             rotate_gt = cv2.warpAffine(val_gt, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
#             rotate_gt = np.rint(rotate_gt)
#             assert np.unique(rotate_gt).shape[0] == 2
#
#             if debug:
#                 cv2.imshow('c2 {}'.format(degree), rotate_img)
#                 cv2.waitKey(100)
#                 cv2.imshow('gt2 {}'.format(degree), rotate_gt)
#                 cv2.waitKey(100)
#
#         # flip image lr, ud
#         im_fliplr = fliplr_aug.augment_image(val_im)
#         gt_fliplr = fliplr_aug.augment_image(val_gt)
#         assert np.unique(gt_fliplr).shape[0] == 2
#
#         im_flipud = flipud_aug.augment_image(val_im)
#         gt_flipud = flipud_aug.augment_image(val_gt)
#         assert np.unique(gt_flipud).shape[0] == 2
#
#         cv2.imshow('lr', im_fliplr)
#         cv2.waitKey(100)
#         cv2.imshow('lr_gt', gt_fliplr)
#         cv2.waitKey(100)
#
#         cv2.imshow('ud', im_flipud)
#         cv2.waitKey(100)
#         cv2.imshow('ud_gt', gt_flipud)
#         cv2.waitKey(100)
# ---------------------------------------

        # cv2.imshow('c', val_im)
        # cv2.waitKey(100)
        # cv2.imshow('d', val_gt)
        # cv2.waitKey(100)

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    checkpoint = ModelCheckpoint('models/' + save_path, monitor='val_iou', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir='./logs/' + save_path[:-3], write_graph=False)
    callbacks_list = [checkpoint, tensorboard]
    steps_per_epoch = (train_df.shape[0] / batch_size) * num_augs
    input_shape = (300,300,3)

    #model = Unet(input_shape)
    model = load_model('models/model_1_1.h5', custom_objects={'f1': f1, 'iou': iou})
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['acc', f1, iou])
    model.fit_generator(gen, epochs=25, steps_per_epoch= int(steps_per_epoch),
                    callbacks=callbacks_list, validation_data=(x_val, y_val))

    model.save('path.h5')


