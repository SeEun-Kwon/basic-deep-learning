import numpy as np
import os
import cv2


def txt_load(path):
    f = open(path, 'r')
    lines = f.readlines()
    txt = np.zeros(shape=(len(lines)), dtype=np.uint8)
    for i in range(len(lines)):
        txt[i] = int(lines[i]) - 1
    return txt, len(lines)


def img_load(path):
    names = os.listdir(path)         # type(names)=list
    img = np.zeros(shape=(len(names), 128, 128, 3), dtype=np.uint8)

    for i in range(len(names)):
        full = path + '/' + names[i]
        img[i, :, :, :] = cv2.imread(full)
        if i % 10000 == 0:
            print(f'{i} / {len(names)}')

    img = np.transpose(img, (0, 3, 1, 2))

    return img


def mini_batch(data_size, batch_size, img, gt):
    idx = np.random.randint(low=0, high=data_size, size=batch_size)
    batch_img = np.zeros(shape=(batch_size, 3, 128, 128), dtype=np.float32)
    batch_gt = np.zeros(shape=(batch_size, 200), dtype=np.float32)

    for i in range(batch_size):
        temp = idx[i]
        aug_flag = np.random.randint(3)

        if aug_flag == 0:
            batch_img[i, :, :, :] = img[temp, :, :, :] / 255 * 2 - 1  # size: (32,128,128,3)
        elif aug_flag == 1:
            batch_img[i, :, :, :] = (cv2.flip(img[temp, :, :, :], 1)) / 255 * 2 - 1         # 상하 반전
        elif aug_flag == 2:
            batch_img[i, 0, :, :] = (cv2.rotate(img[temp, 0, :, :], cv2.ROTATE_90_CLOCKWISE)) / 255 * 2 - 1     # 시계방향 90도 회전
            batch_img[i, 1, :, :] = (cv2.rotate(img[temp, 1, :, :], cv2.ROTATE_90_CLOCKWISE)) / 255 * 2 - 1
            batch_img[i, 2, :, :] = (cv2.rotate(img[temp, 2, :, :], cv2.ROTATE_90_CLOCKWISE)) / 255 * 2 - 1
        batch_gt[i, gt[temp]] = 1

    return batch_img, batch_gt
