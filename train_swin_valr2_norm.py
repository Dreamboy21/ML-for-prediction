import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import swin_t, Swin_T_Weights

import matplotlib.pyplot as plt

# ============================================================
# 固定路径（你不用改）
# ============================================================
CSV_PATH    = r"E:\ML for prediction\data\train\labels.csv"        # CSV: filename,b1,...
IMAGE_ROOT  = r"E:\ML for prediction\data\train\images"            # 图像目录：image_0001.png 等
OUTPUT_PATH = r"E:\ML for prediction\best_swin_regression.pth"     # 最优模型权重（Swin）

# 图像保存目录（自动创建）
PLOT_DIR    = r"E:\ML for prediction\plots_swin"

EPOCHS      = 10
BATCH_SIZE  = 8
LR          = 1e-4
VAL_RATIO   = 0.2
NUM_WORKERS = 0   # Windows 建议 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 标签缩放因子（用于归一化）
TARGET_SCALE = 1e5

# 每条结构有多少 k 点
N_KPOINTS = 31


# ============================================================
# 解析复数字符串，只要实部
# ============================================================
def parse_complex_real(x):
    if isinstance(x, (int, float, np.floating)):
        return float(x)

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    # 去掉 i / j 标记
    s_no_i = s.replace("i", "").replace("I", "").replace("j", "").replace("J", "")

    # 找到中间的 + / - （跳过开头的符号）
    idx = None
    for i, ch in enumerate(s_no_i[1:], start=1):
        if ch in "+-":
            idx = i
            break

    if idx is not None:
        real_str = s_no_i[:idx]
    else:
        real_str = s_no_i

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

    # 解析复数 → 实数
    label_df = df[label_cols].applymap(parse_complex_real)
    # 缺失值填 0
    label_df = label_df.fillna(0.0)

    labels = label_df.values.astype(np.float32)
    labels_norm = labels / TARGET_SCALE  # 归一化

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
# Swin 回归模型
# ============================================================
def create_swin_model(num_targets):
    """
    使用 torchvision 的 Swin-T 预训练权重，
    将分类头替换成回归头，输出 num_targets 维度。
    """
    weights = Swin_T_Weights.IMAGENET1K_V1
    model = swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_targets)
    return model


# ============================================================
# 训练 / 验证
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    count = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

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
    running_loss = 0.0
    count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            count += bs

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / max(count, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return val_loss, all_preds, all_targets


# ============================================================
# 通用评估（用于算 Train/Val 的 R² 等指标）
# ============================================================
def evaluate_for_metrics(model, loader, device):
    """
    在给定 loader 上跑一遍前向，收集所有预测和标签
    用于计算 R² / MAE / RMSE 等指标
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return all_targets, all_preds


# ============================================================
# 指标：R² / MAE / RMSE / MMSE
#   - R²_norm：归一化空间 + flatten
#   - R²_orig：原始尺度 + flatten
# ============================================================
def compute_metrics_all(y_true_norm, y_pred_norm):
    """
    y_true_norm, y_pred_norm: [N, B]，已经 /TARGET_SCALE 过
    返回：
      - *_norm：归一化空间
      - *_orig：原始空间（仅参考）
      - r2_norm：归一化空间 + flatten 的 R²
      - r2：原始空间 + flatten 的 R²
    """
    # ---------- 1) 归一化空间 ----------
    diff_norm = y_pred_norm - y_true_norm  # [N, B]

    mse_norm = float(np.mean(diff_norm ** 2))
    mae_norm = float(np.mean(np.abs(diff_norm)))
    rmse_norm = float(np.sqrt(mse_norm))

    # 每条能带的 MSE（归一化）
    band_mse_norm = np.mean(diff_norm ** 2, axis=0)  # [B]
    mmse_norm = float(np.min(band_mse_norm))         # “论文式” MMSE（取最小那条带）

    # ----- R²_norm（在归一化空间计算）-----
    y_true_norm_flat = y_true_norm.reshape(-1)
    y_pred_norm_flat = y_pred_norm.reshape(-1)

    ss_res_norm = float(np.sum((y_true_norm_flat - y_pred_norm_flat) ** 2))
    ss_tot_norm = float(np.sum((y_true_norm_flat - np.mean(y_true_norm_flat)) ** 2))
    r2_norm = 1.0 - ss_res_norm / ss_tot_norm if ss_tot_norm > 0 else 0.0

    # ---------- 2) 原始物理空间 ----------
    scale = float(TARGET_SCALE)
    y_true = y_true_norm * scale
    y_pred = y_pred_norm * scale
    diff_orig = y_pred - y_true

    mse_orig = float(np.mean(diff_orig ** 2))
    mae_orig = float(np.mean(np.abs(diff_orig)))
    rmse_orig = float(np.sqrt(mse_orig))

    band_mse_orig = band_mse_norm * (scale ** 2)
    mmse_orig = float(np.min(band_mse_orig))

    # ----- R²（原始尺度）-----
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
        "r2_norm": r2_norm,   # 归一化空间 R²
        "r2": r2,             # 原始尺度 R²
    }


# ============================================================
# 在最后一次验证结果上：画一张结构图的 8 条能带对比（使用归一化后的频率）
# 数据布局：每个 k 有 8 个值，一共 31 个 k
# 顺序为 [k0_b1, k0_b2, ..., k0_b8, k1_b1, ..., k30_b8]
# ============================================================
def plot_band_structure_comparison_one_sample(
    y_true_norm_all,
    y_pred_norm_all,
    filenames,
    sample_idx,
    save_dir,
    tag="final",
):
    """
    y_true_norm_all, y_pred_norm_all: [N, B]  (归一化后的所有验证样本)
    filenames: 长度 N，对应每个验证样本的文件名
    sample_idx: 选第几个样本来画图

    数据布局（当前）：
      - 总长度 B = N_KPOINTS * N_BANDS
      - 顺序是：对每个 k，有 N_BANDS 个值
        [k0_b1, k0_b2, ..., k0_b8, k1_b1, k1_b2, ..., k1_b8, ...]
      - 即 shape = (N_KPOINTS, N_BANDS)，行=k点，列=band
      - 画图时我们要变成 (N_BANDS, N_KPOINTS)
    """

    os.makedirs(save_dir, exist_ok=True)

    num_samples, B = y_true_norm_all.shape
    if num_samples == 0:
        print("验证集中没有样本，无法画能带对比图。")
        return

    if sample_idx < 0 or sample_idx >= num_samples:
        print(f"sample_idx={sample_idx} 越界，使用 0 号样本。")
        sample_idx = 0

    if B % N_KPOINTS != 0:
        print(
            f"警告：标签维度 B={B} 不能被 N_KPOINTS={N_KPOINTS} 整除，"
            "无法按 (N_KPOINTS, N_BANDS) 解释，请检查数据格式。"
        )
        return

    N_BANDS = B // N_KPOINTS  # 这里应该是 8

    # ✅ 直接使用归一化后的值作图（不再乘 TARGET_SCALE）
    y_true_flat = y_true_norm_all[sample_idx]  # [B]
    y_pred_flat = y_pred_norm_all[sample_idx]  # [B]

    # 先按「每个 k 有 N_BANDS 个值」解释：
    # shape = (N_KPOINTS, N_BANDS)  -> 行：k，列：band
    y_true_k_band = y_true_flat.reshape(N_KPOINTS, N_BANDS)  # [31, 8]
    y_pred_k_band = y_pred_flat.reshape(N_KPOINTS, N_BANDS)  # [31, 8]

    # 转置成 (N_BANDS, N_KPOINTS) 便于按 band 画线
    y_true_bands = y_true_k_band.T  # [8, 31]
    y_pred_bands = y_pred_k_band.T  # [8, 31]

    k_idx = np.arange(N_KPOINTS)

    fname = str(filenames[sample_idx])
    base = os.path.splitext(os.path.basename(fname))[0]

    plt.figure(figsize=(8, 6))
    for b in range(N_BANDS):
        # 真实：实线（归一化频率/能量）
        plt.plot(k_idx, y_true_bands[b], linestyle="-", alpha=0.8)
        # 预测：虚线（归一化频率/能量）
        plt.plot(k_idx, y_pred_bands[b], linestyle="--", alpha=0.8)

    plt.xlabel("k-point index")
    plt.ylabel("Normalized energy / frequency")
    plt.title(
        f"Normalized band structure comparison ({N_BANDS} bands)\n"
        f"Sample: {base}, tag: {tag}"
    )
    # 图例只说明线型含义
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
# 画 Loss / 指标 曲线（含 Train / Val R²）
# ============================================================
def plot_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epochs = np.arange(1, len(history["train_mse"]) + 1)

    # ---- 1) Loss 曲线（MSE_norm）----
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

    # ---- 2) R² / MAE / RMSE （Val + Train R²）----
    plt.figure()
    # 验证集
    plt.plot(epochs, history["mae_norm"], label="Val MAE_norm")
    plt.plot(epochs, history["rmse_norm"], label="Val RMSE_norm")
    plt.plot(epochs, history["r2_norm"], label="Val R²_norm")
    plt.plot(epochs, history["r2_orig"], label="Val R²_orig")
    # 训练集 R² 用虚线
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

    # ---- 3) 单独 MMSE_norm 曲线 ----
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

    # 1) 读 CSV
    print(f"使用固定路径 CSV: {CSV_PATH}")
    print(f"图片根目录: {IMAGE_ROOT}")
    filenames, labels_norm = load_and_process_csv(CSV_PATH)
    num_samples, num_targets = labels_norm.shape
    print(f"共有样本: {num_samples}，每个样本标签维度: {num_targets}")

    # 2) 划分 train/val
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

    # 3) transform
    train_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageRegressionDataset(
        IMAGE_ROOT, train_filenames, train_labels, transform=train_transform
    )
    val_dataset = ImageRegressionDataset(
        IMAGE_ROOT, val_filenames, val_labels, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 4) Swin 模型 / loss / optimizer
    model = create_swin_model(num_targets=num_targets).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    # 用于画曲线的历史记录
    history = {
        "train_mse": [],
        "val_mse": [],
        "mae_norm": [],
        "rmse_norm": [],
        "mmse_norm": [],
        "r2_norm": [],        # 验证集 R²_norm
        "r2_orig": [],        # 验证集 R²_orig
        "train_r2_norm": [],  # 训练集 R²_norm
        "train_r2_orig": [],  # 训练集 R²_orig
    }

    # 用来保存“最后一次验证”的预测/真实值
    last_val_preds_norm = None
    last_val_targets_norm = None

    # 5) 训练循环
    for epoch in range(1, EPOCHS + 1):
        # 1) 训练一轮
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # 2) 验证
        val_loss, preds_norm_val, targets_norm_val = validate(
            model, val_loader, criterion, DEVICE
        )
        val_metrics = compute_metrics_all(targets_norm_val, preds_norm_val)

        # 3) 在训练集上评估 R² 等指标
        train_targets_norm_all, train_preds_norm_all = evaluate_for_metrics(
            model, train_loader, DEVICE
        )
        train_metrics = compute_metrics_all(train_targets_norm_all, train_preds_norm_all)

        # 记录历史
        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)
        history["mae_norm"].append(val_metrics["mae_norm"])
        history["rmse_norm"].append(val_metrics["rmse_norm"])
        history["mmse_norm"].append(val_metrics["mmse_norm"])
        history["r2_norm"].append(val_metrics["r2_norm"])
        history["r2_orig"].append(val_metrics["r2"])

        history["train_r2_norm"].append(train_metrics["r2_norm"])
        history["train_r2_orig"].append(train_metrics["r2"])

        # 保存这一轮的 val 结果（用作最后一轮的能带图）
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
            f"MMSE_norm={val_metrics['mmse_norm']:.3e} | "
            f"MAE_orig={val_metrics['mae_orig']:.2f} | "
            f"RMSE_orig={val_metrics['rmse_orig']:.2f} | "
            f"MMSE_orig={val_metrics['mmse_orig']:.2e}"
        )

        # 保存最优模型（按 ValMSE_norm）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "num_targets": num_targets,
                    "target_scale": TARGET_SCALE,
                },
                OUTPUT_PATH,
            )
            print(f"  ✅ 保存当前最优模型 → {OUTPUT_PATH}")

    # 画训练过程曲线
    plot_curves(history, PLOT_DIR)

    # =======================================================
    # 用“最后一次验证”的结果，画一张结构图的能带对比图（归一化频率）
    # =======================================================
    if last_val_targets_norm is not None and last_val_preds_norm is not None:
        # 这里选验证集里的第 0 个样本，你也可以改成别的 index
        sample_idx = 0
        plot_band_structure_comparison_one_sample(
            y_true_norm_all=last_val_targets_norm,
            y_pred_norm_all=last_val_preds_norm,
            filenames=val_filenames,
            sample_idx=sample_idx,
            save_dir=PLOT_DIR,
            tag="final_epoch",
        )


if __name__ == "__main__":
    main()
