import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from FCN_network import *
from FCN_function import *
# from FCN_function_bs import load_semantic_seg_data
from torchvision import models
import time
from tqdm import tqdm
import os
import cv2

# data load
print('Datasets loading...')
path = '/home/aivs/바탕화면/hdd/BS/VOC_dataset'

while 1:
    choice = input('from (1) scratch or (2) npy: ')
    if choice == '1':
        train, test, train_gts, test_gts = data_load(path)
        np.save('/home/aivs/바탕화면/hdd/KSE/npy/voc/train.npy', train)
        np.save('/home/aivs/바탕화면/hdd/KSE/npy/voc/test.npy', test)
        np.save('/home/aivs/바탕화면/hdd/KSE/npy/voc/train_gts.npy', train_gts)
        np.save('/home/aivs/바탕화면/hdd/KSE/npy/voc/test_gts.npy', test_gts)
        print('data save finished')
        break
    elif choice == '2':
        train = np.load('/home/aivs/바탕화면/hdd/KSE/npy/voc' + '/train.npy')
        test = np.load('/home/aivs/바탕화면/hdd/KSE/npy/voc' + '/test.npy')
        train_gts = np.load('/home/aivs/바탕화면/hdd/KSE/npy/voc' + '/train_gts.npy')
        test_gts = np.load('/home/aivs/바탕화면/hdd/KSE/npy/voc' + '/test_gts.npy')
        print('Datasets loading completed')
        break
    else:
        print('choose 1 or 2')

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "potted plant", "sheep", "sofa", "train", "tv/monitor"]

VOC_COLORMAP = np.array([[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0], [0, 192, 128], [128, 64, 0]])

train_names = os.listdir(path + '/train/train_gt')
test_names = os.listdir(path + '/test/test_img')
class_num = len(VOC_CLASSES)
train_n = len(train)
test_n = len(test)

batch_size = 32
lr = 0.001
iter = 200000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FCN_8().to(device)
model_load = 190000
model.load_state_dict(torch.load(f'/home/aivs/anaconda3/envs/kse/Segmentation/model/{model_load}.pt'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

start_time = 0
# train
for i in range(190001, iter + 1):
    # learning rate scheduling
    if (i % 50000 == 0) and (i != 0):
        optimizer.param_groups[0]['lr'] /= 2

    model.train()
    optimizer.zero_grad()

    batch_img, batch_gts = mini_batch(train_n, batch_size, train, train_gts)
    batch_img, batch_gts = torch.from_numpy(batch_img), torch.from_numpy(batch_gts)
    batch_img, batch_gts = batch_img.to(device), batch_gts.to(device)
    # batch_img: (32, 3, 256, 256), batch_gt: (32, 1, 256, 256)
    output = model(batch_img)    # output.shape = (32, 21, 256, 256)

    loss = criterion(output, batch_gts)
    loss.backward()
    optimizer.step()

    f1 = open('FCN8_loss.txt', 'a+')
    f2 = open('FCN8 pixel accuracy and mIoU.txt', 'a+')

    # test loss
    if i % 100 == 0:
        model.eval()
        with torch.no_grad():
            batch_img, batch_gts = mini_batch(test_n, batch_size, test, test_gts)
            batch_img, batch_gts = torch.from_numpy(batch_img).to(device), torch.from_numpy(batch_gts).to(device)

            output = model(batch_img)  # output.shape = (32, 21, 256, 256)
            test_loss = criterion(output, batch_gts)
        f1.write(f'{i}\t{loss:.5f}\t{test_loss:.5f}\n')
        # print(f'{i}: {loss.item():.4f}')

    # test and save model
    if (i % 10000 == 0) and (i != 190000):
        if i != 0:
            torch.save(model.state_dict(), f'model/{i}.pt')
        model.eval()

        #initialize
        total_pixel_accuracy, total_mIoU = 0, 0
        total_c_m = np.zeros(shape=(21, 21), dtype=np.uint32)

        with torch.no_grad():
            for j in tqdm(range(test_n), desc='testing...'):
                test_img = test[j:j+1, :, :, :] / 255.0 * 2.0 - 1.0
                test_img = np.transpose(test_img, (0, 3, 1, 2))
                test_img = torch.from_numpy(test_img.astype(np.float32)).to(device)

                test_output = model(test_img)
                test_output = np.squeeze(test_output.cpu().numpy())
                test_output = np.transpose(test_output, (1, 2, 0))   # (256, 256, 21)

                # class만 저장
                test_gt = np.argmax(test_gts[j, :, :, :], axis=-1)      # (256, 256)
                pred = np.argmax(test_output, axis=-1)                  # (256, 256)

                # accuracy
                pixel_accuracy, count = 0, 0
                confusion_mtx = np.zeros(shape=(21, 21), dtype=np.uint32)       # (pred, gt)
                comparison = np.zeros(shape=(2, 256 * 256), dtype=np.uint8)
                comparison[0], comparison[1] = pred.reshape(-1), test_gt.reshape(-1)
                comparison = np.transpose(comparison, (1, 0))
                score = np.zeros(shape=test_n, dtype=np.float32)
                for p in range(21):
                    for q in range(21):
                        count = np.sum(np.all(comparison == (p, q), axis=-1))
                        confusion_mtx[p, q] = count
                        total_c_m[p, q] += count
                        if p == q:
                            pixel_accuracy += count
                total_pixel_accuracy += pixel_accuracy
                pixel_accuracy = pixel_accuracy / (256 * 256) * 100

                count, single_mIoU = 0, 0
                score = np.zeros(shape=21, dtype=np.float32)
                for k in range(21):
                    if sum(confusion_mtx[:, k]) != 0:
                        count += 1
                        union = sum(confusion_mtx[k, :]) + sum(confusion_mtx[:, k]) - confusion_mtx[k, k]
                        single_mIoU += (confusion_mtx[k, k] / union)
                        score[k] = confusion_mtx[k, k] / union
                single_mIoU /= count

                # saving predicted image
                result_img = index_to_rgb(pred)    # index를 rgb로 변환하여 result image 생성, (256, 256, 3)
                os.makedirs(f'/home/aivs/바탕화면/hdd/KSE/FCN_output/{i}', exist_ok=True)
                cv2.imwrite(f'/home/aivs/바탕화면/hdd/KSE/FCN_output/{i}/{test_names[j][:-4]}({pixel_accuracy:.4},{single_mIoU:.4}).jpg', result_img)

        count = 0
        for j in range(class_num):
            total_img_union = sum(total_c_m[j, :]) + sum(total_c_m[:, j]) - total_c_m[j, j]
            if total_img_union != 0:
                total_mIoU += (total_c_m[j, j] / total_img_union)
                count += 1
        total_pixel_accuracy /= (256 * 256 * test_n)
        total_mIoU /= count
        f2.write(f'{i}\t{total_pixel_accuracy:.4}\t{total_mIoU:.4}\n')

        test_time = time.time()
        elapsed_time = (test_time - start_time) / 60
        learning_rate = optimizer.param_groups[0]['lr']
        print(f'step: {i}\t||  pixel accuracy: {total_pixel_accuracy:.4f},\tmIoU: {total_mIoU:.4},\tlr: {learning_rate:.10},\ttime: {elapsed_time:.1}')
    f1.close()
    f2.close()


