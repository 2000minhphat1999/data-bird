"""
Tạo PDF báo cáo BirdCLEF+ 2026 bằng ReportLab + Matplotlib
"""
import os, io, textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Font (fallback sang Helvetica nếu không có DejaVu) ──────────────────────
FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

# Thử load DejaVu để hiện Unicode / tiếng Việt
_DEJAVU = r"C:\Windows\Fonts\DejaVuSans.ttf"
_DEJAVU_B = r"C:\Windows\Fonts\DejaVuSans-Bold.ttf"
if os.path.exists(_DEJAVU):
    pdfmetrics.registerFont(TTFont("DejaVu", _DEJAVU))
    if os.path.exists(_DEJAVU_B):
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", _DEJAVU_B))
        FONT_BOLD = "DejaVu-Bold"
    FONT = "DejaVu"

W, H = A4
BLUE   = colors.HexColor("#1565C0")
LBLUE  = colors.HexColor("#E3F2FD")
GREEN  = colors.HexColor("#2E7D32")
ORANGE = colors.HexColor("#E65100")
GRAY   = colors.HexColor("#616161")
LGRAY  = colors.HexColor("#F5F5F5")

# ── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, fontName=FONT, **kw)

sTitle   = S("sTitle",   fontName=FONT_BOLD, fontSize=24, textColor=BLUE,   alignment=TA_CENTER, spaceAfter=6)
sSub     = S("sSub",     fontName=FONT,      fontSize=12, textColor=GRAY,   alignment=TA_CENTER, spaceAfter=4)
sH1      = S("sH1",      fontName=FONT_BOLD, fontSize=14, textColor=BLUE,   spaceBefore=12, spaceAfter=4)
sH2      = S("sH2",      fontName=FONT_BOLD, fontSize=11, textColor=GREEN,  spaceBefore=8,  spaceAfter=3)
sBody    = S("sBody",    fontName=FONT,      fontSize=9,  leading=13,       alignment=TA_JUSTIFY, spaceAfter=3)
sMono    = S("sMono",    fontName="Courier", fontSize=8,  leading=11,       spaceAfter=2)
sCaption = S("sCaption", fontName=FONT,      fontSize=8,  textColor=GRAY,   alignment=TA_CENTER, spaceAfter=6)

# ── Helper: matplotlib → ReportLab Image ─────────────────────────────────────
def fig_to_image(fig, w_cm=15, h_cm=8):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w_cm*cm, height=h_cm*cm)

# ── Load data ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
df   = pd.read_csv(os.path.join(BASE, "birdclef-2026", "train.csv"))
tax  = pd.read_csv(os.path.join(BASE, "birdclef-2026", "taxonomy.csv"))

# ── Chart 1: Pie — Taxonomy class distribution (clips) ───────────────────────
def chart_class_pie():
    data = df["class_name"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    palette = ["#1565C0","#2E7D32","#E65100","#6A1B9A","#AD1457"]
    wedges, texts, autotexts = ax.pie(
        data.values, labels=data.index, autopct="%1.1f%%",
        colors=palette[:len(data)], startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=9)
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_color("white"); at.set_fontweight("bold")
    ax.set_title("Phân bố clip theo lớp sinh vật", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig

# ── Chart 2: Bar — Rating distribution ───────────────────────────────────────
def chart_rating_bar():
    rc = df["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    bars = ax.bar(rc.index.astype(str), rc.values,
                  color=["#EF5350" if r == 0.0 else "#1565C0" for r in rc.index],
                  edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Rating", fontsize=9)
    ax.set_ylabel("Số clip", fontsize=9)
    ax.set_title("Phân bố Rating chất lượng clip", fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=7)
    ax.axvline(x="3.0", color="green", linestyle="--", linewidth=1.2, alpha=0.7, label="MIN_RATING=3.0")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig

# ── Chart 3: Horizontal bar — Top/Bottom 10 species ─────────────────────────
def chart_species_bar():
    counts = df["primary_label"].value_counts()
    merged = counts.reset_index()
    merged.columns = ["primary_label", "count"]
    merged = merged.merge(tax[["primary_label","common_name"]], on="primary_label", how="left")

    top10 = merged.head(10)
    bot10 = merged.tail(10).iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

    # Top 10
    ax1.barh(top10["common_name"], top10["count"], color="#1565C0", edgecolor="white")
    ax1.set_title("Top 10 loài nhiều clip nhất", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Số clip", fontsize=8)
    ax1.tick_params(axis="y", labelsize=7.5)
    ax1.tick_params(axis="x", labelsize=7.5)
    ax1.grid(axis="x", alpha=0.3)
    for i, v in enumerate(top10["count"]):
        ax1.text(v + 3, i, str(v), va="center", fontsize=7)

    # Bottom 10
    ax2.barh(bot10["common_name"], bot10["count"], color="#E65100", edgecolor="white")
    ax2.set_title("Top 10 loài ít clip nhất", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Số clip", fontsize=8)
    ax2.tick_params(axis="y", labelsize=7.5)
    ax2.tick_params(axis="x", labelsize=7.5)
    ax2.grid(axis="x", alpha=0.3)
    for i, v in enumerate(bot10["count"]):
        ax2.text(v + 0.03, i, str(v), va="center", fontsize=7)

    fig.tight_layout(pad=2)
    return fig

# ── Chart 4: Taxonomy species count bar ─────────────────────────────────────
def chart_taxonomy_bar():
    tc = tax["class_name"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    palette = ["#1565C0","#2E7D32","#E65100","#6A1B9A","#AD1457"]
    bars = ax.bar(tc.index, tc.values, color=palette[:len(tc)], edgecolor="white")
    ax.set_title("Số loài trong taxonomy theo lớp", fontsize=11, fontweight="bold")
    ax.set_ylabel("Số loài", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(int(bar.get_height())), ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig

# ── Table helper ─────────────────────────────────────────────────────────────
def make_table(data, col_widths=None, header=True):
    ts = TableStyle([
        ("FONTNAME",    (0,0), (-1,0),  FONT_BOLD),
        ("FONTNAME",    (0,1), (-1,-1), FONT),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("BACKGROUND",  (0,0), (-1,0),  BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.lightgrey),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0),(-1,-1), 6),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ])
    rows = []
    for row in data:
        rows.append([Paragraph(str(c), sMono) for c in row])
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    t.setStyle(ts)
    return t

# ── BUILD PDF ─────────────────────────────────────────────────────────────────
OUT = os.path.join(BASE, "BirdCLEF2026_Report.pdf")
doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm,
                        topMargin=2*cm, bottomMargin=2*cm)

story = []

# ════════════════════════════════════════════════════════
# TRANG BÌA
# ════════════════════════════════════════════════════════
story.append(Spacer(1, 3*cm))
story.append(Paragraph("BirdCLEF+ 2026", sTitle))
story.append(Paragraph("Báo Cáo Toàn Dự Án", S("sub2", fontName=FONT_BOLD, fontSize=16,
                        textColor=BLUE, alignment=TA_CENTER, spaceAfter=6)))
story.append(HRFlowable(width="80%", thickness=2, color=BLUE, spaceAfter=10))
story.append(Paragraph("Kaggle Competition · Multi-label Sound Event Detection", sSub))
story.append(Paragraph("Ngày tạo: 2026-03-15", sSub))
story.append(Spacer(1, 1*cm))

# Info box
info_data = [
    ["Hạng mục", "Chi tiết"],
    ["Platform",  "Kaggle"],
    ["Task",      "Multi-label SED (Sound Event Detection)"],
    ["Input",     "Soundscape OGG · chunk 5 giây"],
    ["Output",    "Xác suất 234 loài mỗi chunk 5s"],
    ["Metric",    "ROC-AUC (macro)"],
    ["GPU local", "NVIDIA RTX 3050 Ti · 4 GB VRAM"],
    ["OS",        "Windows 11 · Python 3.13 · PyTorch 2.6.0+cu124"],
]
story.append(make_table(info_data, col_widths=[5*cm, 11*cm]))
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph(
    "BirdCLEF+ 2026 mở rộng ra ngoài chim (Aves) sang Amphibia, Insecta, Mammalia và Reptilia. "
    "Data lấy từ iNaturalist thay vì Xeno-canto như các năm trước.",
    sBody))

story.append(PageBreak())

# ════════════════════════════════════════════════════════
# 1. PHÂN TÍCH DỮ LIỆU
# ════════════════════════════════════════════════════════
story.append(Paragraph("1. Phân Tích Dữ Liệu", sH1))
story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

# Tổng quan nhanh
story.append(Paragraph("1.1  Tổng quan", sH2))
summary_data = [
    ["Chỉ số", "Giá trị"],
    ["Tổng train clips",       "35,549"],
    ["Số loài (taxonomy)",     "234"],
    ["Train soundscapes",      "4,961 file OGG"],
    ["Files trên disk",        "35,549 / 35,549  (100%)"],
    ["Sau filter rating≥3.0",  "21,295 clips  (59.9%)"],
    ["Clips/loài (min/max/median)", "1 / 499 / 125"],
]
story.append(make_table(summary_data, col_widths=[8*cm, 8*cm]))
story.append(Spacer(1, 0.4*cm))

# Pie + taxonomy bar
story.append(Paragraph("1.2  Phân bố lớp sinh vật", sH2))
story.append(Paragraph(
    "Aves (chim) chiếm đến 97.9% tổng số clip, trong khi Insecta, Mammalia và Reptilia "
    "cực kỳ ít dữ liệu — đây là thách thức lớn nhất của competition năm nay.", sBody))

fig1 = chart_class_pie()
fig4 = chart_taxonomy_bar()
img1 = fig_to_image(fig1, 7.5, 5.5)
img4 = fig_to_image(fig4, 8.0, 5.5)
story.append(Table([[img1, img4]], colWidths=[8*cm, 9*cm]))
story.append(Paragraph("Trái: Phân bố clip · Phải: Số loài trong taxonomy", sCaption))

# Clips by class table
story.append(Paragraph("1.3  Clips theo lớp", sH2))
class_data = [
    ["Lớp", "Số loài", "Số clip", "Tỷ lệ"],
    ["Aves",      "162", "34,799", "97.9%"],
    ["Amphibia",  " 35",    "451",  "1.3%"],
    ["Insecta",   " 28",    "199",  "0.6%"],
    ["Mammalia",  "  8",     "99",  "0.3%"],
    ["Reptilia",  "  1",      "1",  "0.0%"],
]
story.append(make_table(class_data, col_widths=[4*cm, 4*cm, 4*cm, 4*cm]))
story.append(Spacer(1, 0.5*cm))

# Rating bar
story.append(Paragraph("1.4  Phân bố rating", sH2))
story.append(Paragraph(
    "36.1% clip có rating 0.0 (chất lượng kém). Pipeline lọc MIN_RATING=3.0 "
    "để giữ ~21,295 clip chất lượng tốt hơn.", sBody))
story.append(fig_to_image(chart_rating_bar(), 15, 6))
story.append(Paragraph("Cột đỏ = rating 0.0  ·  Đường xanh lá = ngưỡng lọc MIN_RATING=3.0", sCaption))

# Species bar
story.append(Paragraph("1.5  Top/Bottom 10 loài theo số clip", sH2))
story.append(fig_to_image(chart_species_bar(), 17, 6))
story.append(Paragraph("Imbalance nặng giữa loài nhiều nhất (499) và ít nhất (1 clip)", sCaption))

story.append(PageBreak())

# ════════════════════════════════════════════════════════
# 2. PIPELINE KỸ THUẬT
# ════════════════════════════════════════════════════════
story.append(Paragraph("2. Pipeline Kỹ Thuật", sH1))
story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

story.append(Paragraph("2.1  Audio Preprocessing", sH2))
audio_data = [
    ["Tham số", "Giá trị"],
    ["Sample rate",    "32,000 Hz"],
    ["Chunk length",   "5 giây = 160,000 samples"],
    ["Pad strategy",   "Tile (lặp lại) nếu clip ngắn hơn 5s"],
    ["Crop strategy",  "Random crop trong training"],
]
story.append(make_table(audio_data, col_widths=[7*cm, 9*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("2.2  Mel Spectrogram", sH2))
mel_data = [
    ["Tham số", "Giá trị"],
    ["n_fft",         "1024"],
    ["hop_length",    "320  →  ~100 frames/giây"],
    ["n_mels",        "128"],
    ["fmin / fmax",   "20 Hz / 16,000 Hz"],
    ["Output shape",  "(1, 128, 500)  — 1 channel · 128 mel bins · 500 time steps"],
    ["Backend",       "librosa (ưu tiên) · numpy fallback"],
]
story.append(make_table(mel_data, col_widths=[5*cm, 11*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("2.3  Model Architecture", sH2))
model_data = [
    ["Thành phần", "Chi tiết"],
    ["Backbone",     "EfficientNet-B0 (timm)"],
    ["Input",        "(B, 1, 128, 500)"],
    ["Feature dim",  "1280  (sau GlobalAvgPool)"],
    ["Head",         "Dropout(0.3) → Linear(1280→512) → ReLU → Dropout(0.15) → Linear(512→234)"],
    ["Output",       "(B, 234)  raw logits"],
    ["Loss",         "BCEWithLogitsLoss + pos_weight (xử lý class imbalance)"],
    ["Total params", "4.78M"],
    ["Model size",   "~19 MB (FP32)"],
]
story.append(make_table(model_data, col_widths=[5*cm, 11*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("2.4  Training Config", sH2))
train_data = [
    ["Tham số", "Giá trị"],
    ["Epochs",        "20  (early stopping patience=5)"],
    ["Batch size",    "16  (effective 32 với grad_accum=2)"],
    ["Optimizer",     "AdamW  (lr=1e-3, weight_decay=1e-4)"],
    ["Scheduler",     "Cosine Annealing + 2 epoch warmup  (min_lr=1e-6)"],
    ["AMP",           "Mixed Precision FP16  (tiết kiệm ~50% VRAM)"],
    ["Grad clip",     "1.0"],
    ["Augmentation",  "Mixup (alpha=0.5, p=0.5) + SpecAugment (time=20, freq=10)"],
    ["CV",            "StratifiedKFold 5-fold theo primary_label"],
]
story.append(make_table(train_data, col_widths=[5*cm, 11*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("2.5  Inference", sH2))
infer_data = [
    ["Tham số", "Chi tiết"],
    ["Input",       "Soundscape OGG file"],
    ["Chunking",    "Non-overlapping 5s windows"],
    ["TTA",         "Original + time-flip  (trung bình 2 predictions)"],
    ["Output",      "submission.csv  (row_id × 234 species probabilities)"],
    ["row_id",      "{filename}_{end_second}"],
]
story.append(make_table(infer_data, col_widths=[4*cm, 12*cm]))

story.append(PageBreak())

# ════════════════════════════════════════════════════════
# 3. VRAM BUDGET
# ════════════════════════════════════════════════════════
story.append(Paragraph("3. VRAM Budget — RTX 3050 Ti (4 GB)", sH1))
story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

vram_data = [
    ["Thành phần", "VRAM ước tính"],
    ["EfficientNet-B0 weights",     "~0.5 GB"],
    ["Batch 16 mel-spec (FP16)",    "~0.8 GB"],
    ["Gradients (FP16)",            "~0.5 GB"],
    ["Optimizer states",            "~0.5 GB"],
    ["Buffer",                      "~0.7 GB"],
    ["TỔNG",                        "~3.0 GB  ✓ an toàn"],
]
ts_vram = TableStyle([
    ("FONTNAME",    (0,0),(-1,0),  FONT_BOLD),
    ("FONTNAME",    (0,1),(-1,-1), FONT),
    ("FONTSIZE",    (0,0),(-1,-1), 9),
    ("BACKGROUND",  (0,0),(-1,0),  BLUE),
    ("TEXTCOLOR",   (0,0),(-1,0),  colors.white),
    ("BACKGROUND",  (0,-1),(-1,-1), colors.HexColor("#C8E6C9")),
    ("FONTNAME",    (0,-1),(-1,-1), FONT_BOLD),
    ("ROWBACKGROUNDS",(0,1),(-1,-2),[colors.white, LGRAY]),
    ("GRID",        (0,0),(-1,-1), 0.3, colors.lightgrey),
    ("TOPPADDING",  (0,0),(-1,-1), 5),
    ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING", (0,0),(-1,-1), 8),
])
rows_v = [[Paragraph(str(c), sMono) for c in row] for row in vram_data]
t_vram = Table(rows_v, colWidths=[10*cm, 6*cm])
t_vram.setStyle(ts_vram)
story.append(t_vram)
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    "Nếu OOM → đặt BATCH_SIZE=8 trong config.py. "
    "Nếu RAM hệ thống cạn → NUM_WORKERS=0 trong config.py.", sBody))

# ════════════════════════════════════════════════════════
# 4. MÔI TRƯỜNG
# ════════════════════════════════════════════════════════
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("4. Môi Trường & Dependencies", sH1))
story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

env_data = [
    ["Package", "Version"],
    ["Python",          "3.13.12"],
    ["PyTorch",         "2.6.0+cu124"],
    ["CUDA",            "12.9 (driver 576.52)"],
    ["timm",            "1.0.25"],
    ["librosa",         "0.11.0"],
    ["scikit-learn",    "1.8.0"],
    ["pandas",          "3.0.1"],
    ["numpy",           "2.3.5"],
    ["audiomentations", "0.43.1"],
]
story.append(make_table(env_data, col_widths=[8*cm, 8*cm]))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    "Lưu ý: HuggingFace bị block trong nước — pretrained EfficientNet-B0 không tự tải được. "
    "Cần VPN lần đầu để cache weights vào ~/.cache/huggingface, hoặc đặt PRETRAINED=False.", sBody))

story.append(PageBreak())

# ════════════════════════════════════════════════════════
# 5. HƯỚNG DẪN CHẠY
# ════════════════════════════════════════════════════════
story.append(Paragraph("5. Hướng Dẫn Chạy", sH1))
story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

cmds = [
    ("Phân tích nhẹ (không cần GPU, không train)",
     "python explore.py --skip-plots\npython explore.py"),
    ("Test pipeline",
     "python dataset.py   # Expect: Mel shape: (1, 128, 500)\npython model.py    # Expect: Output: torch.Size([2, 234])"),
    ("Train",
     "python train.py\npython train.py --fold 1"),
    ("Inference",
     "python infer.py --ckpt output/best_fold0.pt"),
]
for title, cmd in cmds:
    story.append(Paragraph(title, sH2))
    story.append(Paragraph(cmd.replace("\n","<br/>"), sMono))
    story.append(Spacer(1, 0.2*cm))

story.append(Paragraph("Output files:", sH2))
out_data = [
    ["File", "Mô tả"],
    ["output/best_fold0.pt",       "Model checkpoint tốt nhất"],
    ["output/history_fold0.csv",   "Training log (loss, AUC theo epoch)"],
    ["output/submission.csv",      "Submission nộp lên Kaggle"],
]
story.append(make_table(out_data, col_widths=[7*cm, 9*cm]))

# ════════════════════════════════════════════════════════
# 6. NEXT STEPS
# ════════════════════════════════════════════════════════
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("6. Next Steps — Nâng Cao Điểm", sH1))
story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

ns_data = [
    ["Mức", "Hướng cải thiện", "Tác động"],
    ["1 — Dễ",  "Tăng lên EfficientNet-B2 nếu VRAM còn dư",              "Nhỏ"],
    ["1 — Dễ",  "Dùng train_soundscapes để validate",                     "Nhỏ"],
    ["1 — Dễ",  "Tăng DURATION từ 5s lên 10s",                           "Nhỏ"],
    ["2 — TB",  "Backbone EfficientAT (pretrained AudioSet)",              "Lớn"],
    ["2 — TB",  "Overlapping chunks (stride 2.5s)",                       "TB"],
    ["2 — TB",  "Pseudo-labeling trên train_soundscapes",                  "TB"],
    ["3 — Khó", "Ensemble 3-5 folds",                                     "Lớn"],
    ["3 — Khó", "Ensemble EfficientNet + EfficientAT + BirdNET",          "Rất lớn"],
    ["3 — Khó", "Threshold tuning per class",                             "TB"],
    ["3 — Khó", "TTA: thêm pitch shift, time stretch",                    "Nhỏ-TB"],
]
story.append(make_table(ns_data, col_widths=[3*cm, 11*cm, 3*cm]))

story.append(Spacer(1, 1*cm))
story.append(HRFlowable(width="100%", thickness=1, color=GRAY, spaceAfter=6))
story.append(Paragraph(
    "BirdCLEF+ 2026 Baseline Report  ·  Tạo tự động bằng ReportLab  ·  2026-03-15",
    S("footer", fontName=FONT, fontSize=8, textColor=GRAY, alignment=TA_CENTER)))

# ── Render ───────────────────────────────────────────────────────────────────
doc.build(story)
print(f"PDF saved: {OUT}")
