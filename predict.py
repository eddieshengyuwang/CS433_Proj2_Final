from keras.models import load_model
import glob
import time
from ml_utils import *
from mask_to_submission import *

def predict(model, training=False, debug=False):
    now = time.time()
    if not training and not debug:
        # save predictions to test_predictions folder
        folder = 'data/test_predictions/{}'.format(now)
        os.mkdir(folder)

    img_dir = 'data/test'
    if training:
        img_dir = 'data/training/images'
    images = sorted(glob.glob(img_dir + '/*.png'),
                    key=lambda x: int(os.path.splitext(x.split('_')[-1])[0]))

    start_1 = time.time()
    for img_path in images:
        print(img_path)
        img = cv2.imread(img_path) / 255
        img_reshape = cv2.resize(img, (300,300))
        img_reshape = img_reshape.reshape((1, *img_reshape.shape))

        start_2 = time.time()
        pred = model.predict(img_reshape).reshape((300,300))
        print('pred time: ', time.time() - start_2)
        pred = cv2.resize(pred, (608, 608))
        pred = (pred > 0.4).astype(np.uint8) * 255

        if debug:
            cv2.imshow('orig', img)
            cv2.waitKey(200)
            cv2.imshow('pred', pred)
            cv2.waitKey(200)
            if training:
                gt_img = img_path.replace('images', 'groundtruth')
                cv2.imshow('gt', cv2.imread(gt_img))
                cv2.waitKey(100)
            input('Press any key to go to the next image: ')

        if not training and not debug:
            # save predictions to test_predictions folder
            cv2.imwrite('{}/{}'.format(folder,img_path.split(os.sep)[-1]), pred)
    print('\n\nTotal execution time: ', time.time() - start_1)

    if not training and not debug:
        return folder


def predict2(model, debug=False):
    now = time.time()
    # save predictions to test_predictions folder
    folder = 'data/test_predictions/{}'.format(now)
    os.mkdir(folder)

    img_dir = 'data/test'
    images = sorted(glob.glob(img_dir + '/*.png'),
                    key=lambda x: int(os.path.splitext(x.split('_')[-1])[0]))

    start_1 = time.time()
    img_reshapes = []
    for img_path in images:
        print(img_path)
        img = cv2.imread(img_path) / 255
        img_reshape = cv2.resize(img, (300,300))
        img_reshapes.append(img_reshape)
    img_reshapes = np.array(img_reshapes)

    start_2 = time.time()
    batch_pred = model.predict(img_reshapes, batch_size=4)
    print('Batch pred time: ', time.time() - start_2)

    for i, pred_img in enumerate(batch_pred):
        pred_img = cv2.resize(pred_img, (608, 608))
        pred_img = (pred_img > 0.5).astype(np.uint8) * 255
        if debug:
            cv2.imshow('orig', cv2.imread(images[i]))
            cv2.waitKey(200)
            cv2.imshow('pred', pred_img)
            cv2.waitKey(200)
            input('Press any key to go to the next image: ')
        cv2.imwrite('{}/{}'.format(folder, images[i].split(os.sep)[-1]), pred_img)

    print('\n\nTotal execution time: ', time.time() - start_1)
    if not debug:
        return folder

if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_name
    training = False
    debug = False
    model = load_model('models/{}'.format(model_path), custom_objects={'iou': iou, 'f1': f1})
    folder = predict(model, training=training, debug=debug) # if debug is True, will return None
    if folder:
        submission_filename = 'submissions/' + model_path[:-3] + ".csv"
        images = sorted(glob.glob(folder + '/*.png'), key=lambda x: int(os.path.splitext(x.split('_')[-1])[0]))
        masks_to_submission(submission_filename, *images)





