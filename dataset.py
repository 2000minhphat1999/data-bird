"""
BirdCLEF+ 2026 — Dataset & DataLoader
- Streaming: đọc audio từng file, không preload vào RAM
- Mel Spectrogram on-the-fly
- SpecAugment + Mixup
"""
import os
import ast
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

try:
    import librosa
    BACKEND = "librosa"
except ImportError:
    import soundfile as sf
    BACKEND = "soundfile"

from config import CFG


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = CFG.SAMPLE_RATE) -> np.ndarray:
    """Load audio file, resample về target sr."""
    if BACKEND == "librosa":
        audio, _ = librosa.load(path, sr=sr, mono=True)
    else:
        audio, orig_sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            # simple resample bằng numpy nếu không có librosa
            ratio = sr / orig_sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio), new_len),
                np.arange(len(audio)),
                audio
            )
    return audio.astype(np.float32)


def audio_to_melspec(audio: np.ndarray) -> np.ndarray:
    """Chuyển audio array → log Mel Spectrogram (H×W float32)."""
    if BACKEND == "librosa":
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=CFG.SAMPLE_RATE,
            n_fft=CFG.N_FFT,
            hop_length=CFG.HOP_LENGTH,
            n_mels=CFG.N_MELS,
            fmin=CFG.FMIN,
            fmax=CFG.FMAX,
        )
        mel = librosa.power_to_db(mel, ref=np.max)
    else:
        # Tính mel-spec thủ công bằng numpy nếu không có librosa
        from numpy.lib.stride_tricks import as_strided
        # STFT
        n_fft = CFG.N_FFT
        hop   = CFG.HOP_LENGTH
        win   = np.hanning(n_fft)
        pad   = np.zeros(n_fft // 2)
        audio = np.concatenate([pad, audio, pad])
        n_frames = 1 + (len(audio) - n_fft) // hop
        frames = as_strided(audio, shape=(n_fft, n_frames),
                            strides=(audio.strides[0], audio.strides[0]*hop))
        spec = np.abs(np.fft.rfft(win[:, None] * frames, axis=0)) ** 2
        # Mel filterbank (approximate)
        n_mels = CFG.N_MELS
        mel_fb = np.zeros((n_mels, n_fft // 2 + 1))
        fmin, fmax = CFG.FMIN, CFG.FMAX
        f_min_mel = 2595 * np.log10(1 + fmin / 700)
        f_max_mel = 2595 * np.log10(1 + fmax / 700)
        mel_points = np.linspace(f_min_mel, f_max_mel, n_mels + 2)
        hz_points  = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / CFG.SAMPLE_RATE).astype(int)
        for m in range(1, n_mels + 1):
            start, center, end = bin_points[m-1], bin_points[m], bin_points[m+1]
            for k in range(start, center):
                if center > start:
                    mel_fb[m-1, k] = (k - start) / (center - start)
            for k in range(center, end):
                if end > center:
                    mel_fb[m-1, k] = (end - k) / (end - center)
        mel = mel_fb @ spec
        mel = 10 * np.log10(mel + 1e-10)
    return mel.astype(np.float32)


def pad_or_crop(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Đảm bảo audio có đúng target_len samples."""
    if len(audio) < target_len:
        # Pad bằng cách lặp lại
        n_repeat = target_len // len(audio) + 1
        audio = np.tile(audio, n_repeat)
    # Random crop
    start = random.randint(0, len(audio) - target_len)
    return audio[start: start + target_len]


def spec_augment(mel: np.ndarray) -> np.ndarray:
    """SpecAugment: che ngẫu nhiên time steps và frequency bins."""
    mel = mel.copy()
    # Time masking
    if CFG.TIME_MASK > 0:
        t = random.randint(0, CFG.TIME_MASK)
        t0 = random.randint(0, max(0, mel.shape[1] - t))
        mel[:, t0: t0 + t] = mel.min()
    # Frequency masking
    if CFG.FREQ_MASK > 0:
        f = random.randint(0, CFG.FREQ_MASK)
        f0 = random.randint(0, max(0, mel.shape[0] - f))
        mel[f0: f0 + f, :] = mel.min()
    return mel


def normalize_mel(mel: np.ndarray) -> np.ndarray:
    """Chuẩn hoá về [0, 1]."""
    mel_min, mel_max = mel.min(), mel.max()
    if mel_max - mel_min > 0:
        mel = (mel - mel_min) / (mel_max - mel_min)
    return mel


# ── Label utilities ───────────────────────────────────────────────────────────

def build_label_map(taxonomy_csv: str):
    """Trả về dict {primary_label: index} và list labels theo thứ tự taxonomy."""
    tax = pd.read_csv(taxonomy_csv)
    label2idx = {row["primary_label"]: i for i, row in tax.iterrows()}
    idx2label = {i: row["primary_label"] for i, row in tax.iterrows()}
    return label2idx, idx2label


def make_label_vector(primary: str, secondary: list,
                      label2idx: dict, n_classes: int) -> np.ndarray:
    """Multi-label one-hot vector."""
    vec = np.zeros(n_classes, dtype=np.float32)
    if primary in label2idx:
        vec[label2idx[primary]] = 1.0
    for s in secondary:
        if s in label2idx:
            vec[label2idx[s]] = 0.5   # soft label cho secondary
    return vec


# ── Dataset ───────────────────────────────────────────────────────────────────

class BirdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2idx: dict,
                 augment: bool = False):
        self.df       = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.augment  = augment
        self.n_classes = CFG.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load audio ──────────────────────────────────────────────────────
        audio_path = os.path.join(CFG.TRAIN_AUDIO, row["filename"])
        try:
            audio = load_audio(audio_path)
        except Exception:
            # File lỗi → trả về silence
            audio = np.zeros(CFG.N_SAMPLES, dtype=np.float32)

        audio = pad_or_crop(audio, CFG.N_SAMPLES)

        # ── Mel Spectrogram ─────────────────────────────────────────────────
        mel = audio_to_melspec(audio)
        mel = normalize_mel(mel)

        # ── Augment ─────────────────────────────────────────────────────────
        if self.augment:
            mel = spec_augment(mel)

        # Shape: (1, n_mels, time) — 1 channel cho CNN
        mel = torch.tensor(mel).unsqueeze(0)

        # ── Labels ──────────────────────────────────────────────────────────
        try:
            secondary = ast.literal_eval(row.get("secondary_labels", "[]"))
        except Exception:
            secondary = []

        label = make_label_vector(
            row["primary_label"], secondary, self.label2idx, self.n_classes
        )
        label = torch.tensor(label)

        return mel, label


# ── Mixup ─────────────────────────────────────────────────────────────────────

def mixup_batch(inputs, targets, alpha=CFG.MIXUP_ALPHA):
    """Mixup augmentation ở batch level."""
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    idx = torch.randperm(batch_size)
    mixed_inputs  = lam * inputs  + (1 - lam) * inputs[idx]
    mixed_targets = lam * targets + (1 - lam) * targets[idx]
    return mixed_inputs, mixed_targets


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_loaders(fold: int = CFG.FOLD):
    """Tạo train/val DataLoader với StratifiedKFold."""
    df  = pd.read_csv(CFG.TRAIN_CSV)
    tax = pd.read_csv(CFG.TAXONOMY)
    label2idx, idx2label = build_label_map(CFG.TAXONOMY)

    # Filter quality
    if CFG.MIN_RATING is not None:
        df = df[df["rating"] >= CFG.MIN_RATING].reset_index(drop=True)
        print(f"After rating filter (>={CFG.MIN_RATING}): {len(df)} clips")

    # Chỉ giữ files thực sự tồn tại trên disk
    df["_path"] = df["filename"].apply(
        lambda f: os.path.join(CFG.TRAIN_AUDIO, f)
    )
    df = df[df["_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"After file existence check: {len(df)} clips")

    # Stratified split theo primary_label
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True,
                          random_state=CFG.SEED)
    df["fold"] = -1
    for f, (_, val_idx) in enumerate(skf.split(df, df["primary_label"])):
        df.loc[val_idx, "fold"] = f

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df   = df[df["fold"] == fold].reset_index(drop=True)
    print(f"Fold {fold}: train={len(train_df)}, val={len(val_df)}")

    train_ds = BirdDataset(train_df, label2idx, augment=True)
    val_ds   = BirdDataset(val_df,   label2idx, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size  = CFG.BATCH_SIZE,
        shuffle     = True,
        num_workers = CFG.NUM_WORKERS,
        pin_memory  = CFG.PIN_MEMORY,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = CFG.BATCH_SIZE * 2,
        shuffle     = False,
        num_workers = CFG.NUM_WORKERS,
        pin_memory  = CFG.PIN_MEMORY,
    )

    return train_loader, val_loader, label2idx, idx2label


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    label2idx, idx2label = build_label_map(CFG.TAXONOMY)
    print(f"Label map: {len(label2idx)} classes")

    df = pd.read_csv(CFG.TRAIN_CSV)
    df["_path"] = df["filename"].apply(lambda f: os.path.join(CFG.TRAIN_AUDIO, f))
    df = df[df["_path"].apply(os.path.exists)].reset_index(drop=True)

    ds = BirdDataset(df.head(10), label2idx, augment=True)
    mel, label = ds[0]
    print(f"Mel shape : {mel.shape}")    # expect (1, 128, 500)
    print(f"Label shape: {label.shape}") # expect (234,)
    print(f"Label sum  : {label.sum()}")
