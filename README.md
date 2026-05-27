# Deep Learning Study Repository

딥러닝 Classification, Segmentation 모델을 직접 구현하고 학습해본 코드
PyTorch 기반으로 주요 CNN 아키텍처들을 구현 및 실험함

---

## 📌 Project Overview

- 대표적인 CNN 기반 딥러닝 모델 구조 이해
- Classification / Segmentation 모델 구현
- 모델별 구조 및 성능 비교
- 데이터 로딩, 학습, 추론 파이프라인 구현

---

# 📂 Implemented Models

## 1. Image Classification

ImageNet2012를 기반으로 구성한 Custom Dataset을 사용
- Number of Classes: 200
- Image Resolution: 128 × 128
- Training Samples: 200,000
- Validation Samples: 1,000

다음 Classification 모델들을 구현 및 학습
- VGG
- ResNet
- DenseNet

### 주요 구현 내용

- DataLoader / Training / Validation Loop 구현
- Accuracy 및 Loss 기록
- 모델 저장 및 불러오기

---

## 2. Image Segmentation

PASCAL VOC 2012 dataset 사용
- Number of Classes: 21
- Training Samples: 1,464
- Validation Samples: 1,449

다음 Segmentation 모델들을 구현 및 학습
- FCN
- U-Net
- PSPNet

### 주요 구현 내용

- DataLoader / Training / Validation Loop 구현
- mIoU / Pixel Accuracy 평가지표 구현
- Mask Visualization
- Accuracy 및 Loss 기록
- 모델 저장 및 불러오기

---


# 📝 What I Learned

- CNN 아키텍처와 태스크별 특징과 차이점 이해
- Residual Connection 및 Dense Connection 구조 학습
- Semantic Segmentation 파이프라인 구현 경험
- 학습 안정화 및 성능 개선 실험

---

# 🔗 References

- VGGNet : https://arxiv.org/pdf/1409.1556
- ResNet : https://arxiv.org/pdf/1512.03385
- DenseNet : https://arxiv.org/pdf/1608.06993
- FCN Paper : https://arxiv.org/pdf/1411.4038
- U-Net Paper : https://arxiv.org/pdf/1505.04597
- PSPNet Paper : https://arxiv.org/pdf/1612.01105
