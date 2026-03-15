# BirdCLEF+ 2026 — Baseline Pipeline

Tối ưu cho: **RTX 3050 Ti (4 GB VRAM) + RAM 8 GB**

## Cấu trúc

```
birdclef_baseline/
├── config.py       ← Tất cả hyperparameters, đổi DATA_DIR ở đây
├── dataset.py      ← Audio loading, mel-spec, DataLoader
├── model.py        ← EfficientNet-B0 + classification head
├── train.py        ← Training loop (AMP + mixup + early stopping)
├── infer.py        ← Inference trên soundscape → submission.csv
└── requirements.txt
```

## Setup

### Bước 1 — Tạo môi trường
```bash
conda create -n birdclef python=3.10
conda activate birdclef
```

### Bước 2 — Cài PyTorch với CUDA
Kiểm tra CUDA version: `nvidia-smi`

```bash
# CUDA 11.8 (RTX 3050 Ti thường dùng driver này)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Nếu CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Bước 3 — Cài dependencies
```bash
pip install -r requirements.txt
```

### Bước 4 — Cấu hình path
Mở `config.py`, sửa:
```python
DATA_DIR = r"C:\Users\tranv\Downloads\birdclef-2026"
OUTPUT_DIR = r"C:\Users\tranv\Downloads\birdclef-2026\output"
```

## Chạy

### Test dataset pipeline
```bash
python dataset.py
# Expect: Mel shape: (1, 128, 500) | Label shape: (234,)
```

### Test model
```bash
python model.py
# Expect: Output: torch.Size([2, 234]) | Params: ~5.3M
```

### Train
```bash
# Fold 0 (mặc định)
python train.py

# Fold cụ thể
python train.py --fold 1
```

Checkpoint lưu tại: `output/best_fold0.pt`
Training log: `output/history_fold0.csv`

### Inference
```bash
python infer.py --ckpt output/best_fold0.pt
# Submission lưu tại: output/submission.csv
```

## VRAM Budget (RTX 3050 Ti = 4 GB)

| Component         | VRAM     |
|-------------------|----------|
| EfficientNet-B0   | ~0.5 GB  |
| Batch 16 mel-spec | ~0.8 GB  |
| Gradients (FP16)  | ~0.5 GB  |
| Optimizer states  | ~0.5 GB  |
| Buffer            | ~0.7 GB  |
| **Total**         | **~3.0 GB** |

Nếu gặp OOM → giảm `BATCH_SIZE = 8` trong `config.py`

## Nếu RAM bị đầy (8 GB)

Giảm trong `config.py`:
```python
NUM_WORKERS = 0    # bỏ multiprocessing hoàn toàn
BATCH_SIZE  = 8
```

## Expected Training Time (RTX 3050 Ti)

- ~8,570 clips trên disk (partial data) → ~15-20 phút/epoch
- Full dataset 35,549 clips → ~60-90 phút/epoch

## Next Steps sau baseline

1. Tăng lên EfficientNet-B2 nếu muốn accuracy cao hơn
2. Thay backbone bằng EfficientAT (pretrained AudioSet) — gap lớn nhất
3. Ensemble nhiều folds
4. Upload lên Kaggle để submit
