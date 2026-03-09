import os
import re
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import swin_t, Swin_T_Weights


# ============================================================
# 路径配置（按你项目；确认 checkpoint 路径是否正确）
# ============================================================
CSV_PATH = r"E:\ML for prediction\data\train\labels.csv"
IMAGE_ROOT = r"E:\ML for prediction\data\train\images"

SWIN_CKPT_PATH = r"E:\ML for prediction\model\best_swin_regression.pth"
CNN_CKPT_PATH  = r"E:\ML for prediction\model\best_shallow_cnn_regression.pth"

OUT_EXCEL_PATH = r"E:\ML for prediction\r2_swin_vs_cnn_1210.xlsx"

VAL_RATIO = 0.2
N_SELECT = 1210
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 复数解析：只取实部
# ============================================================
def parse_complex_real(x):
    if isinstance(x, (int, float, np.floating)):
        return float(x)

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    s_no_i = s.replace("i", "").replace("I", "").replace("j", "").replace("J", "")

    idx = None
    for i, ch in enumerate(s_no_i[1:], start=1):
        if ch in "+-":
            idx = i
            break
    real_str = s_no_i[:idx] if idx is not None else s_no_i

    try:
        return float(real_str)
    except Exception:
        m = re.match(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", s_no_i)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return np.nan
        return np.nan


# ============================================================
# 读取 CSV：返回 filenames, labels_orig（原始尺度）
# ============================================================
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError("CSV 中必须包含 'filename' 列")

    filenames = df["filename"].astype(str).values
    label_cols = [c for c in df.columns if c != "filename"]
    label_df = df[label_cols].applymap(parse_complex_real).fillna(0.0)

    labels_orig = label_df.values.astype(np.float32)
    return filenames, labels_orig


# ============================================================
# R²：flatten 后算（单样本也适用）
# ============================================================
def r2_score_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ============================================================
# 从 image_0035.png -> 35
# ============================================================
def sample_id_from_filename(fname: str) -> int:
    base = os.path.splitext(os.path.basename(fname))[0]  # image_0035
    m = re.search(r"(\d+)$", base)
    if not m:
        return -1
    return int(m.group(1))


# ============================================================
# Swin 模型：替换 head 为回归输出
# ============================================================
def create_swin_model(num_targets: int) -> nn.Module:
    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_targets)
    return model


# ============================================================
# Shallow CNN（与训练一致）
# ============================================================
class ShallowCNNRegressor(nn.Module):
    def __init__(self, num_targets=248):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 112 * 112, 64)
        self.fc2 = nn.Linear(64, num_targets)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ============================================================
# transforms（必须与各自训练一致）
# ============================================================
SWIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

CNN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


# ============================================================
# 单样本推理 + 计时：返回 (pred_orig_vec, time_ms)
# ============================================================
@torch.no_grad()
def predict_one_vector_and_time(model: nn.Module, image_path: str, transform, device, target_scale: float):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # GPU 计时需同步，保证准确
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    pred_norm = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    pred_orig = pred_norm.detach().cpu().numpy().reshape(-1) * float(target_scale)
    time_ms = (t1 - t0) * 1000.0
    return pred_orig.astype(np.float32), float(time_ms)


# ============================================================
# 主流程
# ============================================================
def main():
    # 1) 读 CSV
    print("🔍 Loading CSV ...")
    filenames, labels_orig = load_csv(CSV_PATH)
    num_samples, num_targets = labels_orig.shape
    print(f"✅ CSV loaded: N={num_samples}, B={num_targets}")

    # 2) 按训练逻辑划分 val，并从 val 随机抽 1210
    rng = np.random.default_rng(SEED)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    val_size = int(num_samples * VAL_RATIO)
    val_idx = indices[:val_size]
    val_filenames = filenames[val_idx]
    val_labels_orig = labels_orig[val_idx]

    pick_n = min(N_SELECT, len(val_filenames))
    pick_local = rng.choice(len(val_filenames), size=pick_n, replace=False)

    pick_filenames = val_filenames[pick_local]
    pick_true_orig = val_labels_orig[pick_local]
    print(f"✅ Val size = {len(val_filenames)}, picked = {pick_n}")

    # 3) 加载 Swin checkpoint
    print("🔍 Loading Swin checkpoint ...")
    swin_ckpt = torch.load(SWIN_CKPT_PATH, map_location=DEVICE)
    swin_num_targets = int(swin_ckpt["num_targets"])
    swin_scale = float(swin_ckpt["target_scale"])

    if swin_num_targets != num_targets:
        raise ValueError(f"Swin checkpoint num_targets={swin_num_targets} != CSV labels B={num_targets}")

    swin_model = create_swin_model(swin_num_targets)
    swin_model.load_state_dict(swin_ckpt["model_state_dict"], strict=True)
    swin_model.to(DEVICE).eval()
    print(f"✅ Swin loaded. target_scale={swin_scale:g}")

    # 4) 加载 CNN checkpoint
    print("🔍 Loading CNN checkpoint ...")
    cnn_ckpt = torch.load(CNN_CKPT_PATH, map_location=DEVICE)
    cnn_num_targets = int(cnn_ckpt["num_targets"])
    cnn_scale = float(cnn_ckpt["target_scale"])

    if cnn_num_targets != num_targets:
        raise ValueError(f"CNN checkpoint num_targets={cnn_num_targets} != CSV labels B={num_targets}")

    cnn_model = ShallowCNNRegressor(num_targets=cnn_num_targets)
    cnn_model.load_state_dict(cnn_ckpt["model_state_dict"], strict=True)
    cnn_model.to(DEVICE).eval()
    print(f"✅ CNN loaded. target_scale={cnn_scale:g}")

    # 5) 逐样本推理、算 R²、记录时间，并统计阈值数量
    rows = []
    swin_r2_gt_09999 = 0  # r2 > 0.9999
    swin_r2_gt_0999  = 0  # r2 > 0.999

    for idx, (fname, y_true_vec) in enumerate(zip(pick_filenames, pick_true_orig), start=1):
        image_path = os.path.join(IMAGE_ROOT, fname)
        sid = sample_id_from_filename(fname)

        if not os.path.exists(image_path):
            rows.append((idx, sid, np.nan, np.nan, np.nan, np.nan))
            print(f"[{idx:04d}/{pick_n}] ❌ missing image: {image_path}")
            continue

        # Swin
        y_pred_swin, t_swin_ms = predict_one_vector_and_time(
            swin_model, image_path, SWIN_TRANSFORM, DEVICE, swin_scale
        )
        r2_swin = r2_score_np(y_true_vec, y_pred_swin)

        # CNN
        y_pred_cnn, t_cnn_ms = predict_one_vector_and_time(
            cnn_model, image_path, CNN_TRANSFORM, DEVICE, cnn_scale
        )
        r2_cnn = r2_score_np(y_true_vec, y_pred_cnn)

        # 统计阈值
        if not np.isnan(r2_swin):
            if r2_swin > 0.9999:
                swin_r2_gt_09999 += 1
            if r2_swin > 0.999:
                swin_r2_gt_0999 += 1

        rows.append((idx, sid, float(r2_swin), float(r2_cnn), float(t_swin_ms), float(t_cnn_ms)))

        if idx % 50 == 0 or idx == pick_n:
            print(f"[{idx:04d}/{pick_n}] done...")

    # 6) 导出 Excel（6 列）
    df = pd.DataFrame(
        rows,
        columns=[
            "index",
            "sample_id",
            "swin_r2",
            "cnn_r2",
            "swin_time_ms",
            "cnn_time_ms",
        ],
    )

    out_dir = os.path.dirname(OUT_EXCEL_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 若文件存在且被 Excel/WPS 打开，会报 PermissionError；请先关闭
    if os.path.exists(OUT_EXCEL_PATH):
        os.remove(OUT_EXCEL_PATH)

    df.to_excel(OUT_EXCEL_PATH, index=False)

    # 7) 打印统计结果
    print("====================================")
    print(f"📊 Swin R² > 0.9999 count: {swin_r2_gt_09999} / {len(df)}")
    print(f"📊 Swin R² > 0.999  count: {swin_r2_gt_0999} / {len(df)}")
    print(f"✅ Excel exported: {OUT_EXCEL_PATH}")
    print("====================================")
    print(df.head(10))


if __name__ == "__main__":
    main()
