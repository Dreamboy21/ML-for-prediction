import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt

# ============================================================
# 固定路径（你不用改）
# ============================================================
CSV_PATH    = r"E:\ML for prediction\data\train\labels.csv"        # CSV: filename,b1,...
IMAGE_ROOT  = r"E:\ML for prediction\data\train\images"            # 图像目录：image_0001.png 等
OUTPUT_PATH = r"E:\ML for prediction\best_shallow_cnn_regression.pth"

# 图像保存目录（自动创建）
PLOT_DIR    = r"E:\ML for prediction\plots_shallow_cnn"

# ============================================================
# 按你论文描述的超参数
# ============================================================
EPOCHS      = 100
BATCH_SIZE  = 8
LR          = 1e-3          # ✅ 0.001
VAL_RATIO   = 0.2
NUM_WORKERS = 0             # Windows 建议 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SCALE = 1e5          # 标签归一化（保留你原来）
N_KPOINTS = 31              # 每条结构 k 点数（保留你原来）

# ============================================================
# 解析复数字符串，只要实部
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
        import re
        m = re.match(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", s_no_i)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return np.nan
        return np.nan

# ============================================================
# 读取 CSV + 处理标签（归一化）
# ============================================================
def load_and_process_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError("CSV 中必须包含 'filename' 列！")

    filenames = df["filename"].values
    label_cols = [c for c in df.columns if c != "filename"]
    print(f"检测到标签列数量: {len(label_cols)}，例如：{label_cols[:5]} ...")

    label_df = df[label_cols].applymap(parse_complex_real).fillna(0.0)
    labels = label_df.values.astype(np.float32)
    labels_norm = labels / TARGET_SCALE

    print(
        f"标签原始范围: min={labels.min():.3f}, max={labels.max():.3f}；"
        f"归一化后范围: min={labels_norm.min():.5f}, max={labels_norm.max():.5f}"
    )
    return filenames, labels_norm

# ============================================================
# Dataset
# ============================================================
class ImageRegressionDataset(Dataset):
    def __init__(self, image_root, filenames, labels_norm, transform=None):
        self.image_root = image_root
        self.filenames = filenames
        self.labels = labels_norm.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_root, fname)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图片文件: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        target = torch.from_numpy(self.labels[idx])
        return img, target

# ============================================================
# ✅ 按论文描述的“浅层 CNN”
# 输入：224×224×3
# Conv(3×3, out=8, stride=1, padding=1) -> 224×224×8
# MaxPool(2×2, stride=2) -> 112×112×8
# Flatten -> 8×112×112 = 100352
# FC1: 100352 -> 64
# FC2: 64 -> 248
# ReLU 激活
# ============================================================
class ShallowCNNRegressor(nn.Module):
    def __init__(self, num_targets=248):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )
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
# 训练 / 验证
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, count = 0.0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        count += bs
    return running_loss / max(count, 1)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, count = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            count += bs

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    val_loss = running_loss / max(count, 1)
    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0,))
    all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,))
    return val_loss, all_preds, all_targets

def evaluate_for_metrics(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0,))
    all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,))
    return all_targets, all_preds

# ============================================================
# 指标：R² / MAE / RMSE / MMSE
# ============================================================
def compute_metrics_all(y_true_norm, y_pred_norm):
    diff_norm = y_pred_norm - y_true_norm

    mse_norm = float(np.mean(diff_norm ** 2))
    mae_norm = float(np.mean(np.abs(diff_norm)))
    rmse_norm = float(np.sqrt(mse_norm))

    band_mse_norm = np.mean(diff_norm ** 2, axis=0)
    mmse_norm = float(np.min(band_mse_norm))

    y_true_norm_flat = y_true_norm.reshape(-1)
    y_pred_norm_flat = y_pred_norm.reshape(-1)

    ss_res_norm = float(np.sum((y_true_norm_flat - y_pred_norm_flat) ** 2))
    ss_tot_norm = float(np.sum((y_true_norm_flat - np.mean(y_true_norm_flat)) ** 2))
    r2_norm = 1.0 - ss_res_norm / ss_tot_norm if ss_tot_norm > 0 else 0.0

    scale = float(TARGET_SCALE)
    y_true = y_true_norm * scale
    y_pred = y_pred_norm * scale
    diff_orig = y_pred - y_true

    mse_orig = float(np.mean(diff_orig ** 2))
    mae_orig = float(np.mean(np.abs(diff_orig)))
    rmse_orig = float(np.sqrt(mse_orig))

    band_mse_orig = band_mse_norm * (scale ** 2)
    mmse_orig = float(np.min(band_mse_orig))

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    ss_res = float(np.sum((y_true_flat - y_pred_flat) ** 2))
    ss_tot = float(np.sum((y_true_flat - np.mean(y_true_flat)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "mse_norm": mse_norm,
        "mae_norm": mae_norm,
        "rmse_norm": rmse_norm,
        "mmse_norm": mmse_norm,
        "mse_orig": mse_orig,
        "mae_orig": mae_orig,
        "rmse_orig": rmse_orig,
        "mmse_orig": mmse_orig,
        "r2_norm": r2_norm,
        "r2": r2,
    }

# ============================================================
# 能带对比图（归一化尺度）
# ============================================================
def plot_band_structure_comparison_one_sample(
    y_true_norm_all,
    y_pred_norm_all,
    filenames,
    sample_idx,
    save_dir,
    tag="final",
):
    os.makedirs(save_dir, exist_ok=True)

    num_samples, B = y_true_norm_all.shape
    if num_samples == 0:
        print("验证集中没有样本，无法画能带对比图。")
        return

    if sample_idx < 0 or sample_idx >= num_samples:
        print(f"sample_idx={sample_idx} 越界，使用 0 号样本。")
        sample_idx = 0

    if B % N_KPOINTS != 0:
        print(f"警告：B={B} 不能被 N_KPOINTS={N_KPOINTS} 整除，检查数据格式。")
        return

    N_BANDS = B // N_KPOINTS

    y_true_flat = y_true_norm_all[sample_idx]
    y_pred_flat = y_pred_norm_all[sample_idx]

    y_true_k_band = y_true_flat.reshape(N_KPOINTS, N_BANDS)
    y_pred_k_band = y_pred_flat.reshape(N_KPOINTS, N_BANDS)

    y_true_bands = y_true_k_band.T
    y_pred_bands = y_pred_k_band.T

    k_idx = np.arange(N_KPOINTS)

    fname = str(filenames[sample_idx])
    base = os.path.splitext(os.path.basename(fname))[0]

    plt.figure(figsize=(8, 6))
    for b in range(N_BANDS):
        plt.plot(k_idx, y_true_bands[b], linestyle="-", alpha=0.8)
        plt.plot(k_idx, y_pred_bands[b], linestyle="--", alpha=0.8)

    plt.xlabel("k-point index")
    plt.ylabel("Normalized energy / frequency")
    plt.title(
        f"Normalized band structure comparison ({N_BANDS} bands)\n"
        f"Sample: {base}, tag: {tag}"
    )

    true_line = plt.Line2D([], [], linestyle="-", color="black", label="True (normalized)")
    pred_line = plt.Line2D([], [], linestyle="--", color="black", label="Pred (normalized)")
    plt.legend(handles=[true_line, pred_line], loc="best")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_name = f"band_structure_compare_norm_{tag}_{base}.png"
    out_path = os.path.join(save_dir, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✅ 归一化能带结构对比图已保存到: {out_path}")

# ============================================================
# 曲线图
# ============================================================
def plot_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(1, len(history["train_mse"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_mse"], label="Train MSE_norm")
    plt.plot(epochs, history["val_mse"], label="Val MSE_norm")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized)")
    plt.title("Train / Val MSE (normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["mae_norm"], label="Val MAE_norm")
    plt.plot(epochs, history["rmse_norm"], label="Val RMSE_norm")
    plt.plot(epochs, history["r2_norm"], label="Val R²_norm")
    plt.plot(epochs, history["r2_orig"], label="Val R²_orig")
    plt.plot(epochs, history["train_r2_norm"], label="Train R²_norm", linestyle="--")
    plt.plot(epochs, history["train_r2_orig"], label="Train R²_orig", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("MAE_norm / RMSE_norm / Train & Val R²")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_curve.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["mmse_norm"], label="Val MMSE_norm")
    plt.xlabel("Epoch")
    plt.ylabel("MMSE (normalized)")
    plt.title("Val MMSE_norm over epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mmse_curve.png"), dpi=300)
    plt.close()

    print(f"曲线图已保存到：{save_dir}")

# ============================================================
# main
# ============================================================
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"使用固定路径 CSV: {CSV_PATH}")
    print(f"图片根目录: {IMAGE_ROOT}")
    filenames, labels_norm = load_and_process_csv(CSV_PATH)

    num_samples, num_targets = labels_norm.shape
    print(f"共有样本: {num_samples}，每个样本标签维度: {num_targets}")

    # ✅ 论文里 FC2 输出 248，你这里也应是 248
    if num_targets != 248:
        print(f"⚠️ 提醒：CSV 标签维度是 {num_targets}，但论文 CNN 输出是 248。")
        print("   如果你确实是 31×8=248，请检查 labels.csv 是否列数正确。")

    # 划分 train/val
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    val_size = int(num_samples * VAL_RATIO)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_filenames = filenames[train_idx]
    train_labels = labels_norm[train_idx]
    val_filenames = filenames[val_idx]
    val_labels = labels_norm[val_idx]

    print(f"训练集: {len(train_filenames)}，验证集: {len(val_filenames)}")

    # ✅ 论文只说输入 224×224，这里保持；Normalize 你可保留或去掉
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    train_dataset = ImageRegressionDataset(IMAGE_ROOT, train_filenames, train_labels, transform=transform)
    val_dataset   = ImageRegressionDataset(IMAGE_ROOT, val_filenames, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ✅ 浅层 CNN
    model = ShallowCNNRegressor(num_targets=num_targets).to(DEVICE)

    # ✅ MSE loss + Adam + lr=0.001（论文要求）
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    history = {
        "train_mse": [],
        "val_mse": [],
        "mae_norm": [],
        "rmse_norm": [],
        "mmse_norm": [],
        "r2_norm": [],
        "r2_orig": [],
        "train_r2_norm": [],
        "train_r2_orig": [],
    }

    last_val_preds_norm = None
    last_val_targets_norm = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        val_loss, preds_norm_val, targets_norm_val = validate(model, val_loader, criterion, DEVICE)
        val_metrics = compute_metrics_all(targets_norm_val, preds_norm_val)

        train_targets_norm_all, train_preds_norm_all = evaluate_for_metrics(model, train_loader, DEVICE)
        train_metrics = compute_metrics_all(train_targets_norm_all, train_preds_norm_all)

        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)
        history["mae_norm"].append(val_metrics["mae_norm"])
        history["rmse_norm"].append(val_metrics["rmse_norm"])
        history["mmse_norm"].append(val_metrics["mmse_norm"])
        history["r2_norm"].append(val_metrics["r2_norm"])
        history["r2_orig"].append(val_metrics["r2"])

        history["train_r2_norm"].append(train_metrics["r2_norm"])
        history["train_r2_orig"].append(train_metrics["r2"])

        last_val_preds_norm = preds_norm_val
        last_val_targets_norm = targets_norm_val

        print(
            f"[Epoch {epoch:02d}/{EPOCHS}] "
            f"TrainMSE_norm={train_loss:.3e} | "
            f"ValMSE_norm={val_loss:.3e} | "
            f"TrainR2_norm={train_metrics['r2_norm']:.6f} | "
            f"TrainR2_orig={train_metrics['r2']:.6f} | "
            f"ValR2_norm={val_metrics['r2_norm']:.6f} | "
            f"ValR2_orig={val_metrics['r2']:.6f} | "
            f"MAE_norm={val_metrics['mae_norm']:.3e} | "
            f"RMSE_norm={val_metrics['rmse_norm']:.3e} | "
            f"MMSE_norm={val_metrics['mmse_norm']:.3e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "num_targets": num_targets,
                    "target_scale": TARGET_SCALE,
                    "arch": "ShallowCNN_1conv_1pool_2fc",
                },
                OUTPUT_PATH,
            )
            print(f"  ✅ 保存当前最优浅层 CNN 模型 → {OUTPUT_PATH}")

    plot_curves(history, PLOT_DIR)

    if last_val_targets_norm is not None and last_val_preds_norm is not None:
        plot_band_structure_comparison_one_sample(
            y_true_norm_all=last_val_targets_norm,
            y_pred_norm_all=last_val_preds_norm,
            filenames=val_filenames,
            sample_idx=0,
            save_dir=PLOT_DIR,
            tag="final_epoch",
        )

if __name__ == "__main__":
    main()
