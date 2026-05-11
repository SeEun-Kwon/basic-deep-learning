import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from network import *
from function import index2rgb, color2index
from torchvision import models
import time
from tqdm import tqdm
import os
import cv2

# data load
print('Datasets loading...')
path = '/home/aivs/바탕화면/hdd/BS/VOC_dataset'
try_name = ('FCN')

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
start = 1
loss_epoch = 100
test_epoch = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FCN_8().to(device)
model_load = 190000
model.load_state_dict(torch.load(f'/home/aivs/anaconda3/envs/kse/Segmentation/model/{model_load}.pt'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)


start_time = time.time()

print('train start')

# train
for i in range(start, iter + 1):
    # learning rate scheduling
    if (i % 50000 == 0) and (i != 0):
        optimizer.param_groups[0]['lr'] /= 2

    model.train()
    optimizer.zero_grad()

    batch_img, batch_gts = mini_batch(train_n, batch_size, train, train_gts)
    batch_img, batch_gts = torch.from_numpy(batch_img), torch.from_numpy(batch_gts)     # batch_img: (32, 3, 256, 256), batch_gt: (32, 1, 256, 256)
    batch_img, batch_gts = batch_img.to(device), batch_gts.to(device)

    output = model(batch_img)    # output.shape = (32, 21, 256, 256)

    # out, aux_out = model(batch_img, True)
    # loss = criterion(out, batch_gts) + a * criterion(aux_out, batch_gts)


    loss = criterion(output, batch_gts)
    loss.backward()
    optimizer.step()

    f1 = open(f'loss.txt', 'a+')
    f2 = open(f'{try_name}.txt', 'a+')

    # test loss
    if i % loss_epoch == 0:
        model.eval()
        with torch.no_grad():
            batch_img, batch_gts = mini_batch(test_n, batch_size, test, test_gts)
            batch_img, batch_gts = torch.from_numpy(batch_img).to(device), torch.from_numpy(batch_gts).to(device)

            output = model(batch_img)  # output.shape = (32, 21, 256, 256)
            test_loss = criterion(output, batch_gts)
        f1.write(f'{i}\t{loss:.5f}\t{test_loss:.5f}\n')
        # print(f'{i}: {loss.item():.4f}')

    # test and save model
    if (i % test_epoch == 0) and (i != start):
        if i != 0:
            torch.save(model.state_dict(), f'model/{try_name}_{i}.pt')
        model.eval()

        # initialize
        total_PA, total_mIoU = 0., 0.
        total_c_m = np.zeros(shape=(21, 21), dtype=np.uint32)

        with torch.no_grad():
            for j in tqdm(range(test_n), desc='testing...'):
                test_img = test[j:j+1, :, :, :] / 255.0 * 2.0 - 1.0
                test_img = np.transpose(test_img, (0, 3, 1, 2))
                test_img = torch.from_numpy(test_img.astype(np.float32)).to(device)

                test_output = model(test_img)
                test_output = test_output.cpu().numpy().squeeze()
                test_output = np.argmax(np.transpose(test_output, (1, 2, 0)), axis=2)   # (256, 256)
                
                test_gt = color2index(cv2.cvtColor(test_gts[j, :, :, :] , cv2.COLOR_BGR2RGB))      # (256, 256)

                # accuracy
                single_PA, single_mIoU, count = 0., 0., 0
                single_c_m = np.zeros(shape=(21, 21), dtype=np.uint32)       # (pred, gt)
                IoU = np.zeros(shape=21, dtype=np.float16)

                comparison = np.zeros(shape=(2, 256 * 256), dtype=np.uint8)  
                comparison[0], comparison[1] = pred.reshape(-1), test_gt.reshape(-1)
                comparison = np.transpose(comparison, (1, 0))
                
                for p in range(21):
                    for q in range(21):
                        count = np.sum(np.all(comparison == (p, q), axis=-1))
                        single_c_m[p, q] = count
                        total_c_m[p, q] += count
                        if p == q:
                            single_PA += count

                count = 0
                for k in range(21):
                    if sum(single_c_m[:, k]) != 0:
                        count += 1
                        single_IoU = single_c_m[k, k] / sum(single_c_m[k, :]) + sum(single_c_m[:, k]) - single_c_m[k, k]
                        single_mIoU += single_IoU

                single_PA /= 256 * 256
                single_mIoU /= count

                # saving predicted image
                os.makedirs(f'/home/aivs/바탕화면/hdd/KSE/output/{try_name}', exist_ok=True)
                cv2.imwrite(f'/home/aivs/바탕화면/hdd/KSE/output/{try_name}/{test_names[j][:-4]}({single_PA:.4f},{single_mIoU:.4f}).jpg', index2rgb(pred))

        count = 0
        for j in range(class_num):
            total_union = sum(total_c_m[j, :]) + sum(total_c_m[:, j]) - total_c_m[j, j]
            IoU = total_c_m[j, j] / total_union
            if total_img_union != 0:
                total_mIoU += IoU
                count += 1
                
        total_PA = np.sum(total_c_m.diagonal()) / (256 * 256 * test_n)
        total_mIoU /= count
        
        f2.write(f'{i}\t{total_PA:.4}\t{total_mIoU:.4}\n')

        test_time = time.time()
        elapsed_time = (test_time - start_time) / 60
        learning_rate = optimizer.param_groups[0]['lr']
        
        print(f'step: {i}\t||  '
        f'PA: {total_PA:.4f},\t'
        f'mIoU: {total_mIoU:.4},\t'
        f'lr: {learning_rate:.10},\t'
        f'time: {elapsed_time:.1}')
        
    f1.close()
    f2.close()


