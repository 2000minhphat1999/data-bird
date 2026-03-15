"""
BirdCLEF+ 2026 — Config
Tối ưu cho RTX 3050 Ti (4 GB VRAM) + RAM 8 GB
"""
import os

class CFG:
    # ── Paths ────────────────────────────────────────────────────────────────
    DATA_DIR    = r"C:\Users\tranv\Downloads\birdclef_baseline\birdclef-2026"
    TRAIN_CSV   = os.path.join(DATA_DIR, "train.csv")
    TAXONOMY    = os.path.join(DATA_DIR, "taxonomy.csv")
    TRAIN_AUDIO = os.path.join(DATA_DIR, "train_audio")
    TEST_SOUNDS = os.path.join(DATA_DIR, "test_soundscapes")
    SUB_CSV     = os.path.join(DATA_DIR, "sample_submission.csv")
    OUTPUT_DIR  = r"C:\Users\tranv\Downloads\birdclef_baseline\birdclef-2026\output"

    # ── Audio ────────────────────────────────────────────────────────────────
    SAMPLE_RATE  = 32000          # 32 kHz — standard cho bird audio
    DURATION     = 5              # giây mỗi chunk
    N_SAMPLES    = SAMPLE_RATE * DURATION   # = 160,000 samples

    # ── Mel Spectrogram ──────────────────────────────────────────────────────
    N_FFT        = 1024
    HOP_LENGTH   = 320            # → ~100 frames/giây
    N_MELS       = 128
    FMIN         = 20             # Hz — lọc bỏ hạ âm
    FMAX         = 16000          # Hz — đủ cho chim + côn trùng
    # Output shape: (1, 128, 500) — 1 channel, 128 mels, 500 time steps

    # ── Model ────────────────────────────────────────────────────────────────
    MODEL_NAME   = "efficientnet_b0"   # ~5.3M params, nhẹ nhất
    NUM_CLASSES  = 234                 # tổng số loài trong taxonomy
    PRETRAINED   = True
    IN_CHANNELS  = 1                   # grayscale mel-spec

    # ── Training — tối ưu cho 4 GB VRAM ────────────────────────────────────
    EPOCHS       = 20
    BATCH_SIZE   = 16              # an toàn với 4 GB VRAM + AMP
    NUM_WORKERS  = 2               # thấp để tiết kiệm RAM
    PIN_MEMORY   = True

    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 2
    MIN_LR       = 1e-6

    USE_AMP      = True            # Mixed Precision — tiết kiệm ~50% VRAM
    GRAD_CLIP    = 1.0
    GRAD_ACCUM   = 2               # effective batch = 16×2 = 32

    # ── Data filtering ───────────────────────────────────────────────────────
    MIN_RATING   = 3.0             # bỏ clips chất lượng thấp
    # Set None để dùng toàn bộ data

    # ── Augmentation ─────────────────────────────────────────────────────────
    USE_MIXUP    = True
    MIXUP_ALPHA  = 0.5
    TIME_MASK    = 20              # SpecAugment: che tối đa 20 time steps
    FREQ_MASK    = 10              # che tối đa 10 freq bins

    # ── Cross-validation ─────────────────────────────────────────────────────
    N_FOLDS      = 5
    FOLD         = 0               # train fold này trước

    # ── Reproducibility ──────────────────────────────────────────────────────
    SEED         = 42

    # ── Device ───────────────────────────────────────────────────────────────
    # Tự detect — ưu tiên GPU
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
