from keras.models import load_model
import numpy as np
import glob
import cv2
import time
import os
from ml_utils import *

def predict(model, img_dir, training=False, debug=False):
    images = glob.glob(img_dir + '/*.png')

    for img_path in images:
        img = cv2.imread(img_path) / 255
        if img.shape != (400,400,3):
            img = cv2.resize(img, (400,400))
        img_reshape = img.reshape((1, *img.shape))
        pred = model.predict(img_reshape).reshape((400,400))
        pred = (pred > 0.3).astype(np.uint8) * 255

        if debug:
            cv2.imshow('orig', img)
            cv2.waitKey(100)
            cv2.imshow('pred', pred)
            cv2.waitKey(100)
            if training:
                gt_img = img_path.replace('images', 'groundtruth')
                cv2.imshow('gt', cv2.imread(gt_img))
                cv2.waitKey(100)
            input('Press any key to go to the next image: ')

        if not training and not debug:
            # save predictions to test_predictions folder
            now = time.time()
            folder = 'data/test_predictions/{}'.format(now)
            os.mkdir(folder)
            cv2.imwrite('{}/{}'.format(folder,img_path.split(os.sep)[-1]), pred)

if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_name
    model = load_model('models/{}'.format(model_path), custom_objects={'iou': iou, 'f1': f1})
    img_dir = 'data/test'
    img_dir = 'data/training/images'
    predict(model, img_dir, training=True, debug=True)


