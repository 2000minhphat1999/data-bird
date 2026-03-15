"""
BirdCLEF+ 2026 — Lightweight EDA & Pipeline Check
Chạy nhanh để kiểm tra data, visualize mel-spec, test model shape.
Không train, không cần GPU.
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import CFG
from dataset import load_audio, audio_to_melspec, normalize_mel, build_label_map, BirdDataset

# ── 1. Dataset Stats ──────────────────────────────────────────────────────────

def show_dataset_stats():
    print("\n" + "="*60)
    print("  1. DATASET STATS")
    print("="*60)

    df  = pd.read_csv(CFG.TRAIN_CSV)
    tax = pd.read_csv(CFG.TAXONOMY)

    print(f"Train CSV       : {len(df):,} rows")
    print(f"Taxonomy classes: {len(tax)}")

    # Class distribution
    class_counts = tax["class_name"].value_counts()
    print(f"\nClass breakdown (taxonomy):")
    for cls, cnt in class_counts.items():
        print(f"  {cls:<12} {cnt:>4} species")

    # Ratings
    print(f"\nRating distribution:")
    print(df["rating"].value_counts().sort_index().to_string())

    # Files on disk
    df["_path"] = df["filename"].apply(lambda f: os.path.join(CFG.TRAIN_AUDIO, f))
    exists = df["_path"].apply(os.path.exists)
    print(f"\nFiles on disk   : {exists.sum():,} / {len(df):,}")

    # Per-label count
    label_counts = df["primary_label"].value_counts()
    print(f"\nSamples per species — min: {label_counts.min()}, "
          f"max: {label_counts.max()}, median: {label_counts.median():.0f}")

    return df[exists].reset_index(drop=True), tax


# ── 2. Visualize Mel Spectrograms ─────────────────────────────────────────────

def show_spectrograms(df, n=6):
    print("\n" + "="*60)
    print("  2. MEL SPECTROGRAM SAMPLES")
    print("="*60)

    sample = df.sample(min(n, len(df)), random_state=42)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    axes = axes.flatten()

    for i, (_, row) in enumerate(sample.iterrows()):
        path = os.path.join(CFG.TRAIN_AUDIO, row["filename"])
        try:
            audio = load_audio(path)
            # crop 5s
            n_samp = CFG.N_SAMPLES
            if len(audio) >= n_samp:
                start = random.randint(0, len(audio) - n_samp)
                audio = audio[start: start + n_samp]
            else:
                audio = np.tile(audio, n_samp // len(audio) + 1)[:n_samp]

            mel = audio_to_melspec(audio)
            mel = normalize_mel(mel)

            axes[i].imshow(mel, origin="lower", aspect="auto", cmap="magma")
            axes[i].set_title(f"{row['common_name']}\n({row['primary_label']})",
                              fontsize=8)
            axes[i].axis("off")
            print(f"  [{i+1}] {row['common_name']} | mel shape: {mel.shape}")
        except Exception as e:
            axes[i].set_title(f"Error: {e}", fontsize=7)
            print(f"  [{i+1}] ERROR: {e}")

    plt.suptitle("Sample Mel Spectrograms (5s chunks)", fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "spectrograms.png")
    plt.savefig(out, bbox_inches="tight", dpi=120)
    print(f"\n  Saved → {out}")
    plt.show()


# ── 3. Class Distribution Plot ────────────────────────────────────────────────

def show_class_distribution(df, tax):
    print("\n" + "="*60)
    print("  3. CLASS DISTRIBUTION")
    print("="*60)

    # Merge label counts with taxonomy
    counts = df["primary_label"].value_counts().reset_index()
    counts.columns = ["primary_label", "count"]
    merged = counts.merge(tax[["primary_label", "class_name", "common_name"]],
                          on="primary_label", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: top-30 species
    top30 = merged.head(30)
    axes[0].barh(top30["common_name"].fillna(top30["primary_label"].astype(str)),
                 top30["count"], color="steelblue")
    axes[0].set_xlabel("Number of clips")
    axes[0].set_title("Top 30 species by clip count")
    axes[0].invert_yaxis()

    # Right: distribution by class_name
    by_class = merged.groupby("class_name")["count"].sum().sort_values(ascending=False)
    axes[1].bar(by_class.index, by_class.values, color="coral")
    axes[1].set_xlabel("Taxonomic class")
    axes[1].set_ylabel("Total clips")
    axes[1].set_title("Clips by taxonomic class")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "class_distribution.png")
    plt.savefig(out, bbox_inches="tight", dpi=120)
    print(f"  Saved → {out}")
    plt.show()


# ── 4. Model Architecture Check ───────────────────────────────────────────────

def check_model():
    print("\n" + "="*60)
    print("  4. MODEL ARCHITECTURE CHECK (CPU)")
    print("="*60)
    import torch
    from model import BirdCLEFModel

    model = BirdCLEFModel(pretrained=False).cpu()
    x = torch.randn(2, 1, CFG.N_MELS, 500)
    with torch.no_grad():
        out = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Input shape  : {list(x.shape)}")
    print(f"  Output shape : {list(out.shape)}   (expect [2, {CFG.NUM_CLASSES}])")
    print(f"  Total params : {total_params/1e6:.2f}M")
    print(f"  Trainable    : {trainable/1e6:.2f}M")

    # Estimate model size
    size_mb = total_params * 4 / 1e6
    print(f"  Model size   : ~{size_mb:.0f} MB (FP32)")


# ── 5. Audio Duration Stats ───────────────────────────────────────────────────

def show_audio_stats(df, n_sample=200):
    print("\n" + "="*60)
    print(f"  5. AUDIO DURATION STATS (sample {n_sample} files)")
    print("="*60)

    sample = df.sample(min(n_sample, len(df)), random_state=42)
    durations = []
    for _, row in sample.iterrows():
        path = os.path.join(CFG.TRAIN_AUDIO, row["filename"])
        try:
            audio = load_audio(path)
            durations.append(len(audio) / CFG.SAMPLE_RATE)
        except Exception:
            pass

    if durations:
        durations = np.array(durations)
        print(f"  Min    : {durations.min():.1f}s")
        print(f"  Max    : {durations.max():.1f}s")
        print(f"  Mean   : {durations.mean():.1f}s")
        print(f"  Median : {np.median(durations):.1f}s")
        print(f"  < 5s   : {(durations < 5).sum()} files ({(durations<5).mean()*100:.1f}%)")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(durations, bins=40, color="teal", edgecolor="white")
        ax.axvline(5, color="red", linestyle="--", label="5s chunk")
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count")
        ax.set_title(f"Audio Duration Distribution (n={len(durations)})")
        ax.legend()
        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "duration_dist.png")
        plt.savefig(out, bbox_inches="tight", dpi=120)
        print(f"  Saved → {out}")
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plots", action="store_true",
                        help="Chỉ in stats, không show/save plots")
    args = parser.parse_args()

    df, tax = show_dataset_stats()

    if not args.skip_plots:
        show_spectrograms(df)
        show_class_distribution(df, tax)
        show_audio_stats(df)

    check_model()

    print("\n" + "="*60)
    print("  DONE — Pipeline ready to train!")
    print("="*60)
