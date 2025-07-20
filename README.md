# 2025 FastMRI Challenge - Enhanced MRI Reconstruction

2025 SNU FastMRI challenge with VarNet + Triple Input NAFNet pipeline

## 프로젝트 개요

이 프로젝트는 FastMRI 데이터를 이용한 MRI 이미지 재구성 성능 향상을 목표로 합니다. 기존 VarNet에 Triple Input NAFNet을 추가하여 다음과 같은 입력을 활용해 최종 재구성 성능을 개선합니다:

1. **VarNet 재구성 이미지**: 기본 재구성 네트워크의 출력
2. **GRAPPA 이미지**: image_grappa 키에서 로드되는 GRAPPA 재구성 이미지  
3. **입력 이미지**: image_input 키에서 로드되는 aliased 입력 이미지

최종 목표는 image_label 키의 alias-free 정답 이미지와 최대한 유사한 재구성을 생성하는 것입니다.

## 파이프라인 아키텍처

```
K-space → VarNet → Reconstructed Image ┐
                                       ├→ Triple Input NAFNet → Final Enhanced Reconstruction
GRAPPA Image ──────────────────────────┤
Aliased Input Image ────────────────────┘
```

## 1. 폴더 계층

### 폴더의 전체 구조
![image](docs/fastmri_folder_structure.png)
* `FastMRI_challenge`, `Data`, `result` 폴더가 위의 구조대로 설정되어 있어야 default argument를 활용할 수 있습니다.
* 본 github repository는 `FastMRI_challenge` 폴더입니다.
* `Data` 폴더는 MRI data 파일을 담고 있으며 아래에 상세 구조를 첨부하겠습니다.
* `result` 폴더는 학습한 모델의 weights을 기록하고 validation, leaderboard dataset의 reconstruction image를 저장하는데 활용되며, 아래에 상세 구조를 첨부하겠습니다.

### Data 폴더의 구조
![image](docs/fastmri_data_structure.png)

#### 데이터 키 구조
각 h5 파일에는 다음과 같은 키들이 포함되어 있습니다:
- `kspace`: 원본 k-space 데이터
- `mask`: 샘플링 마스크
- `image_label`: alias-free 정답 이미지 (학습 목표)
- `image_grappa`: GRAPPA 재구성 이미지 (Triple Input NAFNet 입력)
- `image_input`: aliased 입력 이미지 (Triple Input NAFNet 입력)
- `max`: 정규화를 위한 최대값

* train, val:
    * train, val 폴더는 각각 모델을 학습(train), 검증(validation)하는데 사용하며 각각 image, kspace 폴더로 나뉩니다.
    * 참가자들은 generalization과 representation의 trade-off를 고려하여 train, validation의 set을 자유로이 나눌 수 있습니다.
    * image와 kspace 폴더에 들어있는 파일의 형식은 다음과 같습니다: {brain 또는 knee}\_{mask 형식}\_{순번}.h5
    * ex) brain_acc8_3.h5, knee_acc4_10.h5  
    * {mask 형식}은 "acc4", "acc8" 중 하나입니다.
    * "acc4"의 경우 {순번}은 1 ~ 100, "acc8"의 경우 {순번}은 1 ~ 100 사이의 숫자입니다. 
* leaderboard:
   * **leaderboard는 성능 평가를 위해 활용하는 dataset이므로 절대로 학습 과정에 활용하면 안됩니다.**
   * leaderboard 폴더는 mask 형식에 따라서 acc4과 acc8 폴더로 나뉩니다.
   * acc4과 acc8 폴더는 각각 image, kspace 폴더로 나뉩니다.
   * image와 kspace 폴더에 들어있는 파일의 형식은 다음과 같습니다: {brain 또는 knee}\_{mask 형식}\_{순번}.h5
   * {순번}은 1 ~ 29 사이의 숫자입니다. 

### result 폴더의 구조
* result 폴더는 모델의 이름에 따라서 여러 폴더로 나뉠 수 있습니다.
* 각 모델 폴더는 아래 3개의 폴더로 구성되어 있습니다.
  * checkpoints - `model.pt`, `best_model.pt`의 정보가 있습니다. 모델의 weights 정보를 담고 있습니다.
  * reconstructions_val - validation dataset의 reconstruction을 저장합니다.
  * reconstructions_leaderboard - leaderboard dataset의 reconstruction을 저장합니다.
  * val_loss_log.npy - epoch별로 validation loss를 기록합니다.

## 2. 학습 및 실행 방법

### Step 1: VarNet 학습
먼저 기본 VarNet 모델을 학습합니다:

```bash
# VarNet 학습
bash train_varnet.sh
```

### Step 2: Triple Input NAFNet 학습  
VarNet이 학습된 후, Triple Input NAFNet을 학습합니다:

```bash
# Triple Input NAFNet 학습 (VarNet frozen)
bash train_triple_input_nafnet.sh
```

### Step 3: 추론 실행
학습된 모델들로 최종 재구성을 생성합니다:

```bash
# Triple Input NAFNet으로 향상된 재구성 생성
bash reconstruct_triple_input_nafnet.sh
```

## 3. 프로젝트 구조

```bash
├── .gitignore
├── leaderboard_eval.py
├── README.md
├── reconstruct.py
├── requirements.txt
├── train_varnet.py                          # VarNet 학습 스크립트
├── train_varnet.sh                          # VarNet 학습 쉘 스크립트
├── train_triple_input_nafnet.py             # Triple Input NAFNet 학습 스크립트
├── train_triple_input_nafnet.sh             # Triple Input NAFNet 학습 쉘 스크립트
├── reconstruct_triple_input_nafnet.py       # Triple Input NAFNet 추론 스크립트
├── reconstruct_triple_input_nafnet.sh       # Triple Input NAFNet 추론 쉘 스크립트
├── tutorial.ipynb
└── utils
│   ├── common
│   │   ├── loss_function.py
│   │   └── utils.py
│   ├── data
│   │   ├── load_data.py                     # 기본 데이터 로더
│   │   ├── triple_input_load_data.py        # Triple Input NAFNet용 데이터 로더
│   │   └── transforms.py
│   ├── learning
│   │   ├── train_part.py                    # VarNet 학습 함수
│   │   ├── train_nafnet.py                  # 기본 NAFNet 학습 함수
│   │   └── train_triple_input_nafnet.py     # Triple Input NAFNet 학습 함수
│   └── model
│       ├── varnet.py                        # VarNet 모델
│       ├── unet.py                          # U-Net 모델
│       ├── nafnet.py                        # 기본 NAFNet 모델
│       ├── triple_input_nafnet.py           # Triple Input NAFNet 모델
│       └── fastmri/                         # FastMRI 라이브러리 모듈
```

## 4. 주요 특징

### Triple Input NAFNet
- **3개 입력 채널**: VarNet 재구성 + GRAPPA 이미지 + 입력 이미지
- **Skip Connection**: VarNet 출력을 기반으로 한 skip connection으로 안정적인 학습
- **Frozen VarNet**: 기 학습된 VarNet을 고정하여 NAFNet만 학습

### 데이터 로더 개선
- **자동 GRAPPA 이미지 로드**: `image_grappa` 키에서 자동 로드
- **자동 입력 이미지 로드**: `image_input` 키에서 자동 로드  
- **Fallback 메커니즘**: 키가 없는 경우 k-space에서 자동 생성

### 향상된 성능
- **다중 정보 활용**: 여러 재구성 정보를 동시에 활용하여 성능 개선
- **Weighted SSIM Loss**: 슬라이스 위치에 따른 가중치 적용
- **실시간 비교**: 학습 중 VarNet 대비 성능 개선 정도 실시간 확인

## 5. 모델 성능 비교

학습 및 검증 과정에서 다음과 같은 메트릭들이 출력됩니다:
- VarNet SSIM: 기본 VarNet의 SSIM 점수
- Triple Input NAFNet Enhanced SSIM: 향상된 모델의 SSIM 점수  
- Improvement: 개선 정도

## 6. 체크포인트 및 결과

### 모델 체크포인트
- `/root/result/test_varnet/checkpoints/best_model.pt`: VarNet 모델
- `/root/result/triple_input_nafnet/checkpoints/best_model.pt`: Triple Input NAFNet 모델

### 재구성 결과  
- `/root/result/triple_input_reconstructions/enhanced/`: 최종 향상된 재구성
- `/root/result/triple_input_reconstructions/varnet_only/`: VarNet 재구성 (비교용)

이 파이프라인을 통해 기존 VarNet 대비 향상된 MRI 재구성 성능을 달성할 수 있습니다.
│   │   ├── test_part.py
│   │   └── train_part.py
│   └── model
│       └── varnet.py
└── result
```

## 3. Before you start
* ```train.py```, ```reconstruct.py```, ```leaderboard_eval.py``` 순으로 코드를 실행하면 됩니다.
* ```train.py```
   * train/validation을 진행하고 학습한 model의 결과를 result 폴더에 저장합니다.
   * 가장 성능이 좋은 모델의 weights을 ```best_model.pt```으로 저장합니다. 
* ```reconstruct.py```
   * ```train.py```으로 학습한 ```best_model.pt```을 활용해 leader_board dataset을 reconstruction하고 그 결과를 result 폴더에 저장합니다.
   * Inference Time이 대회 GPU 기준으로 3600초를 초과할 경우 Total SSIM을 기록할 수 없습니다. 실제 Evaluation 때 조교가 확인할 예정이며, inference time은 과도하게 모델이 크지 않는다면 걱정하실 필요 없습니다.
      * 3600초는 reconstruction process의 total time입니다. leaderboard_data --> (optional) preprocessed_data --> reconstruction하는 모든 과정이 포함되며, 여러 개의 모델을 사용하셔서 Inference를 진행하시는 경우에도 한 개의 모델이 아닌 전체 모델에 관하여 3600초를 초과해서는 안 됩니다.
* ```leaderboard_eval.py```
   * ```reconstruct.py```을 활용해 생성한 reconstruction의 SSIM을 측정합니다.
   * SSIM (acc4): acc4 데이터에 대한 reconstruction의 SSIM을 측정합니다.
   * SSIM (acc8): acc8 데이터에 대한 reconstruction의 SSIM을 측정합니다.
   * Total SSIM은 SSIM (acc4), SSIM (acc8)의 평균으로 계산됩니다. Report할 때 이 값을 제출하시면 됩니다.

## 4. How to set?
(python 3.12.9)
```bash
pip3 install -r requirements.txt
```

## 5. How to train?
```bash
python train.py // sh train.sh
```
- validation할 때, reconstruction data를 ```result/reconstructions_val/```에 저장합니다.
- epoch 별로 validation dataset에 대한 loss를 기록합니다.
- sh train.sh를 사용하여도 같은 결과를 얻으실 수 있습니다. Hyperparameter를 쉽게 조작할 수 있습니다.
- **seed 고정**을 하여 이후에 Re-training하였을 때 **같은 결과가 나와야 합니다**.

## 6. How to reconstruct?
```bash
python reconstruct.py // sh reconstruct.sh
```
- leaderboard 평가를 위한 reconstruction data를 ```result/reconstructions_leaderboard```에 저장합니다.

## 7. How to evaluate LeaderBoard Dataset?
```bash
python leaderboard_eval.py // sh leaderboard_eval.sh
```
- leaderboard 순위 경쟁을 위한 4X sampling mask, 8X sampling mask에 대한 SSIM 값을 한번에 구합니다.
- Total SSIM을 제출합니다.

## 8. What to submit!
- github repository(코드 실행 방법 readme에 상세 기록)
- loss 그래프 혹은 기록
- 모델 weight file
- 모델 설명 ppt
