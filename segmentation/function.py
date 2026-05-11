import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from click.core import batch
from numpy.ma.core import shape

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "potted plant", "sheep", "sofa", "train", "tv/monitor"]

VOC_COLORMAP = np.array([[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0], [0, 192, 128], [128, 64, 0]])
# temp = VOC_COLORMAP[:, 0].copy()
# VOC_COLORMAP[:, 0] = VOC_COLORMAP[:, 2]
# VOC_COLORMAP[:, 2] = temp

def data_load(path):
    train_img_names = os.listdir(path + '/train/train_img')
    train_gt_names = os.listdir(path + '/train/train_gt')
    test_img_names = os.listdir(path + '/test/test_img')
    test_gt_names = os.listdir(path + '/test/test_gt')

    trains = np.zeros(shape=(len(train_img_names), 256, 256, 3), dtype=np.uint8)
    tests = np.zeros(shape=(len(test_img_names), 256, 256, 3), dtype=np.uint8)
    train_gts = np.zeros(shape=(len(train_gt_names), 256, 256, 21), dtype=np.uint8)
    test_gts = np.zeros(shape=(len(test_gt_names), 256, 256, 21), dtype=np.uint8)

    for i in range(len(train_img_names)):
        train = cv2.imread(path + '/train/train_img/' + train_img_names[i])
        train_gt = cv2.imread(path + '/train/train_gt/' + train_gt_names[i])
        train = cv2.resize(train, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        train_gt = cv2.resize(train_gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        trains[i, :, :, :] = train
        train_gts[i, :, :, :] = rgb_to_index(train_gt)
    print(f'train data load finished')

    for i in range(len(test_img_names)):
        test = cv2.imread(path + '/test/test_img/' + test_img_names[i])
        test_gt = cv2.imread(path + '/test/test_gt/' + test_gt_names[i])
        test = cv2.resize(test, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        test_gt = cv2.resize(test_gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        tests[i, :, :, :] = test
        test_gts[i, :, :, :] = rgb_to_index(test_gt)
    print(f'test data load finished')

    return trains, tests, train_gts, test_gts

def One_Hot_Encoding(array):    # (b, h, w, c)
    new_array = np.zeros(shape=(array.shape[0], array.shape[1], array.shape[2], 21), dtype=np.uint8)
    for i in range(array.shape[0]):
        new_array[i, :, :, array[i, :, :]] = 1
    return new_array

def mini_batch_bs(data_size, batch_size, img, gt):
    idx = np.random.randint(low=0, high=data_size, size=batch_size)
    batch_img = np.zeros(shape=(batch_size, 256, 256, 3), dtype=np.float32)
    batch_gt = np.zeros(shape=(batch_size, 256, 256, 1), dtype=np.float32)

    for i in range(batch_size):
        temp = idx[i]
        batch_img[i, :, :, :] = (img[temp, :, :, :] / 255.0) * 2.0 - 1.0
        batch_gt[i, :, :, ] = gt[temp, :, :, 0:1]

    batch_img = np.transpose(batch_img, axes=[0, 3, 1, 2])
    batch_gt = np.transpose(batch_gt, axes=[0, 3, 1, 2])

    return batch_img, batch_gt


def mini_batch(data_size, batch_size, img, gt):
    idx = np.random.randint(low=0, high=data_size, size=batch_size)
    batch_img = np.zeros(shape=(batch_size, 256, 256, 3), dtype=np.float32)
    batch_gt = np.zeros(shape=(batch_size, 256, 256, 21), dtype=np.float32)

    for i in range(batch_size):
        temp = idx[i]
        aug_flag = np.random.randint(3)

        if aug_flag == 0:
            batch_img[i, :, :, :] = img[temp, :, :, :] / 255.0 * 2 - 1.0
            batch_gt[i, :, :, :] = gt[temp, :, :, :]
        elif aug_flag == 1:
            batch_img[i, :, :, :] = (cv2.flip(img[temp, :, :, :], 1)) / 255.0 * 2 - 1.0
            batch_gt[i, :, :, :] = cv2.flip(gt[temp, :, :, :], 1)
        elif aug_flag == 2:
            M = cv2.getRotationMatrix2D((256/2, 256/2), 45, 1)
            batch_img[i, :, :, :] = cv2.warpAffine(img[temp, :, :, :], M, (256, 256)) / 255.0 * 2 - 1.0
            batch_gt[i, :, :, :] = cv2.warpAffine(gt[temp, :, :, :], M, (256, 256))

    batch_img = np.transpose(batch_img, (0, 3, 1, 2))
    batch_gt = np.transpose(batch_gt, (0, 3, 1, 2))

    return batch_img, batch_gt

def rgb_to_index(img):
    image = np.zeros(shape=(256, 256, 21), dtype=np.uint8)
    for idx, bgr in enumerate(VOC_COLORMAP):
        where = np.array(np.where(np.all(img == bgr, axis=-1)))  # np.all: train_gt = rgb인 픽셀에 True 반환
        image[where[0], where[1], idx] = 1                         # np.where: True인 요소의 위치 튜플로 반환 (x축 끼리, y축 끼리)
    return image

def index_to_rgb(img):
    output = np.zeros(shape=(256, 256, 3), dtype=np.uint8)
    for i in range(len(VOC_COLORMAP)):
        wh = np.column_stack(np.where(img == i))        # np.column_stack: 여러 vector 입력받아 각 배열을 열(column)로 하는 2차원 배열 생성
        if wh is not None:                              # np.where: 조건 만족하는 index 튜플로 저장 row 0에 모든 index의 row만, row 1에 column만 저장
            for x, y in wh:
                output[x, y] = VOC_COLORMAP[i]
        temp = 0
    return output
