import h5py

data_path = '/root/Data/train/kspace/brain_acc4_1.h5'
    
with h5py.File(data_path, 'r') as f:
    # 최상위 키들만
    print("Top-level keys:", list(f.keys()))
    
    # 각 키의 정보도 함께
    for key in f.keys():
        print(f"{key}: {f[key].shape if hasattr(f[key], 'shape') else type(f[key])}")