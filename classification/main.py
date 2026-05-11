import numpy as np
from function import *
from resnet_network import *
import torch
import torch.optim as optim      # 최적화 알고리즘을 포함하는 모듈
import torch.nn as nn
import time


print('Datasets loading...')
path = 'C:/dataset/tinyImageNet'

# train_gt, train_n = txt_load(path + '/train_gt.txt')
# test_gt, test_n = txt_load(path + '/test_gt.txt')
# train_img, test_img = img_load(path + '/train'), img_load(path + '/test')
#
# np.save(path + '/train_img.npy', train_img)
# np.save(path + '/test_img.npy', test_img)
# np.save(path + '/train_gt.npy', train_gt)
# np.save(path + '/test_gt.npy', test_gt)

train_gt = np.load('C:/dataset/tinyImageNet/npy/train_gt.npy')
test_gt = np.load('C:/dataset/tinyImageNet/npy/test_gt.npy')
train_img = np.load('C:/dataset/tinyImageNet/npy/train_img.npy')
test_img = np.load('C:/dataset/tinyImageNet/npy/test_img.npy')
train_n = len(train_gt)
test_n = len(test_gt)

print('Datasets loading completed')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 텐서 생성할 때 사용할 디바이스 지정

batch_size = 32
iter = 200001
learning_rate = 0.01
num_classes = 200


model = ResNet50().to(device)
loss = nn.CrossEntropyLoss()
# model.load_state_dict(torch.load('C:/PycharmCode/KSE/resnet152_10000.pt'))
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)  # 논문 지시

start_time = time.time()

# train
for i in range(iter):
    if i == 50000:
        optimizer.param_groups[0]['lr'] = 0.005
    elif i == 75000:
        optimizer.param_groups[0]['lr'] = 0.0025
    elif i == 100000:
        optimizer.param_groups[0]['lr'] = 0.001
    elif i == 125000:
        optimizer.param_groups[0]['lr'] = 0.0005
    elif i == 150000:
        optimizer.param_groups[0]['lr'] = 0.0001
    elif i == 175000:
        optimizer.param_groups[0]['lr'] = 0.00001


    model.train()
    optimizer.zero_grad()

    # 미니 배치 실행
    batch_img, batch_gt = mini_batch(train_n, batch_size, train_img, train_gt)
    batch_img, batch_gt = torch.from_numpy(batch_img), torch.from_numpy(batch_gt)
    batch_img, batch_gt = batch_img.to(device), batch_gt.to(device)

    train_output = model(batch_img)
    train_loss = loss(train_output, batch_gt.float())
    train_loss.backward()
    optimizer.step()            # backward 하면서 수집된 gradient로 parameter 조절

    if i % 100 == 0:
        f = open('ResNet50_loss.txt', 'a+')
        f.write(f'{i}\t{train_loss:.4f}\n')
        f.close()
        print(f'{i}: {train_loss.item():.4f}')

    # test
    if (i % 5000 == 0) and (i != 0):    #   5000번에 한번씩 모델 저장, 정확도 테스트
        torch.save(model.state_dict(), f'C:\\Users\\user\\KSE\\pythonProject\\model\\resnet152_{i}.pt')   # 매개변수만 저장
        print('Testing...')
        model.eval()

    # - accuracy 측정
        correct = 0

        with torch.no_grad():
            for j in range(test_img.shape[0]):
                img = test_img[j:j+1, :, :, :] / 255.0 * 2.0 - 1.0
                img = torch.from_numpy(img.astype(np.float32))
                img = img.to(device)
                test_output = model(img)
                pred = test_output.cpu().numpy()
                pred = np.reshape(pred, (1, num_classes))
                if np.argmax(pred) == test_gt[j]:
                    correct += 1

        accuracy = correct / test_n * 100

        stop_time = time.time()
        elapsed_time = (stop_time - start_time) // 60

        print(f'{accuracy:.4f}\t{elapsed_time}')

        f = open('ResNet50_accuracy.txt', 'a+')
        f.write(f'{i}\t{accuracy:4f}\n')
        f.close()
