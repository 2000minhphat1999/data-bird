"""
BirdCLEF+ 2026 — Model
EfficientNet-B0 với 1-channel mel-spec input
Backbone từ timm (pretrained ImageNet)
"""
import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from config import CFG


class BirdCLEFModel(nn.Module):
    """
    Mel Spectrogram → EfficientNet-B0 → Multi-label classifier (234 classes)

    1-channel grayscale mel-spec được replicate thành 3 channels
    để tận dụng pretrained ImageNet weights.
    """

    def __init__(self,
                 model_name: str  = CFG.MODEL_NAME,
                 num_classes: int = CFG.NUM_CLASSES,
                 pretrained: bool = CFG.PRETRAINED,
                 drop_rate: float = 0.3):

        super().__init__()
        self.num_classes = num_classes

        if not HAS_TIMM:
            raise ImportError("pip install timm")

        # Load backbone — in_chans=1 để nhận grayscale mel-spec trực tiếp
        self.backbone = timm.create_model(
            model_name,
            pretrained    = pretrained,
            in_chans      = CFG.IN_CHANNELS,   # 1 channel
            num_classes   = 0,                 # bỏ head gốc
            global_pool   = "avg",
        )

        # Lấy feature dim của backbone
        dummy = torch.zeros(1, CFG.IN_CHANNELS, CFG.N_MELS, 500)
        with torch.no_grad():
            feat_dim = self.backbone(dummy).shape[-1]

        print(f"Backbone: {model_name} | Feature dim: {feat_dim}")

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate / 2),
            nn.Linear(512, num_classes),
        )

        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, time) float32
        Returns:
            logits: (B, num_classes) — raw scores, dùng BCE loss
        """
        features = self.backbone(x)    # (B, feat_dim)
        logits   = self.head(features) # (B, num_classes)
        return logits


# ── Loss ─────────────────────────────────────────────────────────────────────

class BCEWithLogitsLoss(nn.Module):
    """
    BCE loss cho multi-label classification.
    pos_weight để handle class imbalance.
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)


def compute_pos_weight(df, label2idx: dict, n_classes: int,
                       device: str = "cpu") -> torch.Tensor:
    """
    Tính pos_weight = (n_neg / n_pos) per class để cân bằng imbalance.
    Clip ở max=10 để tránh extreme values.
    """
    import ast, numpy as np, pandas as pd

    counts = np.zeros(n_classes)
    total  = len(df)

    for _, row in df.iterrows():
        if row["primary_label"] in label2idx:
            counts[label2idx[row["primary_label"]]] += 1

    pos   = np.clip(counts, 1, None)
    neg   = total - pos
    weight = neg / pos
    weight = np.clip(weight, 1.0, 10.0)
    return torch.tensor(weight, dtype=torch.float32).to(device)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = BirdCLEFModel().to(CFG.DEVICE)
    x = torch.randn(2, 1, CFG.N_MELS, 500).to(CFG.DEVICE)
    out = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")  # expect (2, 234)

    # VRAM usage estimate
    if CFG.DEVICE == "cuda":
        print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {total_params/1e6:.1f}M")
