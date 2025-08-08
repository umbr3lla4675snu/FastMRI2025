# Augmentation Debug Visualization Guide

이 가이드는 `DataTransform`에서 augmentation이 올바르게 적용되고 있는지 시각적으로 확인하는 방법을 설명합니다.

## 기능 설명

`DataTransform` 클래스에 다음 기능들이 추가되었습니다:

1. **디버그 이미지 저장**: augmentation이 적용될 때마다 원본과 변환된 이미지를 비교할 수 있는 시각화 이미지를 저장
2. **제한된 샘플링**: 너무 많은 이미지가 생성되지 않도록 최대 저장 개수 제한
3. **상세한 정보 표시**: k-space와 target의 shape 변화, 차이점 등을 시각적으로 표시

## 사용법

### 1. DataTransform 인스턴스 생성 시 디버그 모드 활성화

```python
from utils.data.transforms import DataTransform

# 디버그 모드로 DataTransform 생성
transform = DataTransform(
    isforward=False,
    max_key='max_reconstruction_target',
    args=your_args,
    debug_augmentation=True,    # 디버그 모드 활성화
    debug_max_samples=10        # 최대 10개 샘플만 저장
)
```

### 2. 기존 코드 수정 예시

**train_triple_input_nafnet.py** 또는 다른 학습 스크립트에서:

```python
# 기존 코드
train_transform = DataTransform(False, 'max_reconstruction_target', args)

# 디버그 모드로 변경
train_transform = DataTransform(
    False, 
    'max_reconstruction_target', 
    args,
    debug_augmentation=True,
    debug_max_samples=20  # 처음 20개 샘플만 저장
)
```

### 3. 테스트 스크립트 실행

```bash
cd /root/FastMRI2025
python debug_augmentation_test.py
```

## 저장되는 이미지 설명

디버그 이미지는 `/root/FastMRI2025/debug_augmentation/` 폴더에 저장됩니다.

각 이미지는 2x4 그리드로 구성되어 있습니다:

**상단 행 (K-space)**:
- 원본 K-space (log scale)
- 변환된 K-space (log scale)  
- K-space 차이점
- Shape 정보

**하단 행 (Target Image)**:
- 원본 Target 이미지
- 변환된 Target 이미지
- Target 차이점
- Shape 정보

## 파일명 규칙

`augmentation_debug_{filename}_slice_{slice_number}.png`

예: `augmentation_debug_test_file_0_slice_0.png`

## 주의사항

1. **성능 영향**: 디버그 모드는 이미지 저장으로 인해 학습 속도가 느려질 수 있습니다.
2. **디스크 공간**: 이미지 파일들이 디스크 공간을 차지합니다.
3. **제한된 샘플링**: `debug_max_samples` 파라미터로 저장할 이미지 수를 제한하는 것을 권장합니다.

## 실제 학습에서 사용하기

실제 학습 시에는 epoch 초기에만 몇 개 샘플을 확인하고 싶다면:

```python
# epoch 0에서만 디버그 이미지 저장
debug_mode = (epoch == 0)

train_transform = DataTransform(
    False, 
    'max_reconstruction_target', 
    args,
    debug_augmentation=debug_mode,
    debug_max_samples=5  # 5개만 저장
)
```

이렇게 하면 augmentation이 제대로 작동하는지 시각적으로 확인할 수 있습니다!
