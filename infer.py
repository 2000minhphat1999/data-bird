"""
BirdCLEF+ 2026 — Inference Script
Soundscape → chunk 5s → mel-spec → model → submission.csv
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast

from config import CFG
from dataset import load_audio, audio_to_melspec, normalize_mel, build_label_map
from model import BirdCLEFModel


@torch.no_grad()
def predict_soundscape(model, audio_path: str,
                        label2idx: dict, idx2label: dict) -> pd.DataFrame:
    """
    Đọc 1 soundscape file, chunk 5s, predict từng chunk.
    Returns DataFrame với row_id và probability mỗi class.
    """
    model.eval()

    # Load full soundscape
    try:
        audio = load_audio(audio_path, sr=CFG.SAMPLE_RATE)
    except Exception as e:
        print(f"  Error loading {audio_path}: {e}")
        return pd.DataFrame()

    filename = os.path.splitext(os.path.basename(audio_path))[0]
    n_samples = CFG.N_SAMPLES   # samples per 5s chunk
    step_size = n_samples       # non-overlapping
    total_len = len(audio)

    rows = []
    offset = 0
    while offset + n_samples <= total_len:
        chunk = audio[offset: offset + n_samples]
        offset += step_size
        end_sec = offset // CFG.SAMPLE_RATE   # label as end second

        # Mel spec
        mel = audio_to_melspec(chunk)
        mel = normalize_mel(mel)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        mel_tensor = mel_tensor.to(CFG.DEVICE)

        # TTA: original + time-flip
        preds_list = []
        for tta_mel in [mel_tensor, mel_tensor.flip(-1)]:
            with autocast(enabled=CFG.USE_AMP):
                logits = model(tta_mel)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds_list.append(probs)
        probs_avg = np.mean(preds_list, axis=0)   # average TTA

        row = {"row_id": f"{filename}_{end_sec}"}
        for idx, label in idx2label.items():
            row[label] = float(probs_avg[idx])
        rows.append(row)

    return pd.DataFrame(rows)


def run_inference(checkpoint_path: str, output_path: str = None):
    """
    Chạy inference trên tất cả test soundscapes.
    Lưu submission.csv ra output_path.
    """
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(CFG.OUTPUT_DIR, "submission.csv")

    # ── Load model ────────────────────────────────────────────────────────────
    label2idx, idx2label = build_label_map(CFG.TAXONOMY)

    model = BirdCLEFModel().to(CFG.DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=CFG.DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val AUC = {ckpt.get('val_auc', 0):.4f})")

    # ── Load submission template để lấy đúng column order ─────────────────
    sub_template = pd.read_csv(CFG.SUB_CSV, nrows=1)
    expected_cols = list(sub_template.columns)   # row_id + 234 species

    # ── Find test soundscapes ─────────────────────────────────────────────
    sound_files = (
        glob.glob(os.path.join(CFG.TEST_SOUNDS, "*.ogg")) +
        glob.glob(os.path.join(CFG.TEST_SOUNDS, "*.wav"))
    )
    print(f"Found {len(sound_files)} test soundscape(s)")

    if len(sound_files) == 0:
        print("⚠ No test soundscapes found.")
        print("  Tạo dummy submission với uniform probabilities...")
        sub = pd.read_csv(CFG.SUB_CSV)
        sub.to_csv(output_path, index=False)
        print(f"Saved dummy submission → {output_path}")
        return

    # ── Predict ───────────────────────────────────────────────────────────────
    all_rows = []
    for i, sf_path in enumerate(sorted(sound_files)):
        print(f"  [{i+1}/{len(sound_files)}] {os.path.basename(sf_path)}")
        df_pred = predict_soundscape(model, sf_path, label2idx, idx2label)
        if not df_pred.empty:
            all_rows.append(df_pred)

    if not all_rows:
        print("No predictions generated.")
        return

    submission = pd.concat(all_rows, ignore_index=True)

    # Reorder columns theo submission template
    for col in expected_cols:
        if col not in submission.columns:
            submission[col] = 1.0 / CFG.NUM_CLASSES   # fallback uniform
    submission = submission[expected_cols]

    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved → {output_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head(3))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(CFG.OUTPUT_DIR, "best_fold0.pt"),
                        help="Path to checkpoint file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output submission CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        print("Chạy train.py trước để tạo checkpoint.")
    else:
        run_inference(args.ckpt, args.output)
