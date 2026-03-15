"""
BirdCLEF+ 2026 — Training Script
- Mixed Precision (AMP) để tiết kiệm VRAM trên RTX 3050 Ti
- Gradient Accumulation để simulate batch lớn hơn
- CosineAnnealing LR scheduler
- Early stopping
- Checkpoint tốt nhất theo val ROC-AUC
"""
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

from config import CFG
from dataset import get_loaders, mixup_batch
from model import BirdCLEFModel, BCEWithLogitsLoss, compute_pos_weight


# ── Reproducibility ───────────────────────────────────────────────────────────
def seed_everything(seed: int = CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Scheduler: Cosine với warmup ──────────────────────────────────────────────
def get_scheduler(optimizer, n_warmup: int, n_total: int):
    def lr_lambda(step):
        if step < n_warmup:
            return step / max(1, n_warmup)
        progress = (step - n_warmup) / max(1, n_total - n_warmup)
        return CFG.MIN_LR / CFG.LR + (1 - CFG.MIN_LR / CFG.LR) * \
               0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_auc(targets: np.ndarray, preds: np.ndarray) -> float:
    """
    Macro ROC-AUC — bỏ qua classes không có positive trong batch.
    """
    scores = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            try:
                scores.append(roc_auc_score(targets[:, i], preds[:, i]))
            except Exception:
                pass
    return float(np.mean(scores)) if scores else 0.0


# ── Train one epoch ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, criterion,
                scheduler, epoch: int) -> dict:
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad()

    for step, (mels, labels) in enumerate(loader):
        mels   = mels.to(CFG.DEVICE, non_blocking=True)
        labels = labels.to(CFG.DEVICE, non_blocking=True)

        # Mixup
        if CFG.USE_MIXUP and random.random() > 0.5:
            mels, labels = mixup_batch(mels, labels)

        # Forward với AMP
        with autocast(enabled=CFG.USE_AMP):
            logits = model(mels)
            loss   = criterion(logits, labels)
            loss   = loss / CFG.GRAD_ACCUM   # normalize cho gradient accum

        # Backward
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (step + 1) % CFG.GRAD_ACCUM == 0:
            if CFG.GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * CFG.GRAD_ACCUM
        steps += 1

        if step % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch} | Step {step}/{len(loader)} | "
                  f"Loss {total_loss/steps:.4f} | LR {lr:.2e}")

    return {"loss": total_loss / steps}


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, criterion) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for mels, labels in loader:
        mels   = mels.to(CFG.DEVICE, non_blocking=True)
        labels = labels.to(CFG.DEVICE, non_blocking=True)

        with autocast(enabled=CFG.USE_AMP):
            logits = model(mels)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    auc = compute_auc(all_labels, all_preds)

    return {
        "loss": total_loss / len(loader),
        "auc":  auc,
        "preds":  all_preds,
        "labels": all_labels,
    }


# ── Main training loop ────────────────────────────────────────────────────────
def train(fold: int = CFG.FOLD):
    seed_everything()
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BirdCLEF+ 2026 — Training  |  Fold {fold}  |  {CFG.DEVICE}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, label2idx, _ = get_loaders(fold)

    # ── Model ────────────────────────────────────────────────────────────────
    model = BirdCLEFModel().to(CFG.DEVICE)

    # ── Loss — với pos_weight cho class imbalance ─────────────────────────
    df = pd.read_csv(CFG.TRAIN_CSV)
    pos_w = compute_pos_weight(df, label2idx, CFG.NUM_CLASSES, CFG.DEVICE)
    criterion = BCEWithLogitsLoss(pos_weight=pos_w)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = CFG.LR,
        weight_decay = CFG.WEIGHT_DECAY,
    )

    # ── Scheduler ────────────────────────────────────────────────────────────
    total_steps  = len(train_loader) // CFG.GRAD_ACCUM * CFG.EPOCHS
    warmup_steps = len(train_loader) // CFG.GRAD_ACCUM * CFG.WARMUP_EPOCHS
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

    # ── AMP Scaler ───────────────────────────────────────────────────────────
    scaler = GradScaler(enabled=CFG.USE_AMP)

    # ── Training ─────────────────────────────────────────────────────────────
    best_auc     = 0.0
    patience     = 5
    no_improve   = 0
    history      = []

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        print(f"\n── Epoch {epoch}/{CFG.EPOCHS} ─────────────────────")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, criterion, scheduler, epoch
        )
        val_metrics = val_epoch(model, val_loader, criterion)

        elapsed = time.time() - t0
        print(f"\n  Train Loss : {train_metrics['loss']:.4f}")
        print(f"  Val   Loss : {val_metrics['loss']:.4f}")
        print(f"  Val   AUC  : {val_metrics['auc']:.4f}")
        print(f"  Time       : {elapsed:.0f}s")

        # VRAM usage
        if CFG.DEVICE == "cuda":
            vram = torch.cuda.memory_allocated() / 1e9
            vram_peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  VRAM       : {vram:.2f} GB (peak {vram_peak:.2f} GB)")

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss":   val_metrics["loss"],
            "val_auc":    val_metrics["auc"],
        })

        # ── Save best ────────────────────────────────────────────────────────
        if val_metrics["auc"] > best_auc:
            best_auc   = val_metrics["auc"]
            no_improve = 0
            ckpt_path = os.path.join(
                CFG.OUTPUT_DIR, f"best_fold{fold}.pt"
            )
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_auc":    best_auc,
                "label2idx":  label2idx,
            }, ckpt_path)
            print(f"  ✓ Saved best model → {ckpt_path}")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")

        # ── Early stopping ───────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best AUC: {best_auc:.4f}")
            break

    # ── Save history ─────────────────────────────────────────────────────────
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(
        os.path.join(CFG.OUTPUT_DIR, f"history_fold{fold}.csv"), index=False
    )
    print(f"\nTraining done. Best val AUC: {best_auc:.4f}")
    return best_auc


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=CFG.FOLD)
    args = parser.parse_args()

    # Kiểm tra GPU trước khi train
    print(f"Device : {CFG.DEVICE}")
    if CFG.DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {total_vram:.1f} GB")
        if total_vram < 4.0:
            print("⚠ VRAM < 4 GB — giảm BATCH_SIZE xuống 8 trong config.py")

    train(fold=args.fold)
