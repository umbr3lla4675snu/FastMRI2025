import h5py
import numpy as np
import matplotlib.pyplot as plt

def show_reconstruction(file_path):
    """
    H5 파일에서 'reconstruction' 데이터셋을 읽어 이미지로 표시합니다.
    """
    try:
        with h5py.File(file_path, 'r') as hf:
            # 파일에 'reconstruction' 키가 있는지 확인
            if 'reconstruction' not in hf:
                print(f"오류: 파일에 'reconstruction' 데이터셋이 없습니다.")
                # 파일의 최상위 키 목록을 출력하여 사용자가 확인할 수 있도록 돕습니다.
                print(f"사용 가능한 키: {list(hf.keys())}")
                return

            # 'reconstruction' 데이터셋을 numpy 배열로 불러오기
            # [:]를 사용하여 파일에서 메모리로 데이터를 완전히 로드합니다.
            recon_data = hf['reconstruction'][:]

            # 데이터 처리
            # 데이터가 3D (slices, height, width) 형태일 경우, 중앙 슬라이스를 선택합니다.
            if recon_data.ndim == 3:
                print(f"3D 데이터 감지 (슬라이스, 높이, 너비). 중앙 슬라이스를 표시합니다.")
                middle_slice_index = recon_data.shape[0] // 2
                display_data = recon_data[middle_slice_index, :, :]
            else:
                display_data = recon_data

            # 데이터가 복소수(complex) 형태일 경우, 절댓값을 취해 실수(magnitude)로 변환합니다.
            if np.iscomplexobj(display_data):
                print("복소수 데이터 감지. 절댓값을 취하여 이미지로 변환합니다.")
                display_data = np.abs(display_data)

            # 이미지 표시
            plt.figure(figsize=(8, 8))
            plt.imshow(display_data, cmap='gray') # 흑백 이미지로 표시
            plt.title('Reconstruction Image')
            plt.axis('off') # 축 정보 숨기기
            plt.show()

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
    except Exception as e:
        print(f"이미지를 표시하는 중 오류가 발생했습니다: {e}")

# --- 실행 부분 ---
# 분석할 H5 파일 경로를 지정하세요.
# 이전 질문의 파일 경로를 예시로 사용합니다.
file_to_show = '/root/result/test_Varnet/reconstructions_val/brain_acc8_99.h5'

# 함수 호출
show_reconstruction(file_to_show)