import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ================================
# 你的训练脚本里用到的复数解析（这里直接复用）
# 如果你已有 train_swin.py 中的 parse_complex_real，也可以改成 from train_swin import parse_complex_real
# ================================
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


# ================================
# 基本参数（按你的路径）
# ================================
CSV_PATH = r"E:\ML for prediction\data\train\labels.csv"
IMAGE_ROOT = r"E:\ML for prediction\data\train\images"

# ✅ 你的 shallow CNN 最优模型路径
CHECKPOINT_PATH = r"E:\ML for prediction\best_shallow_cnn_regression.pth"

OUT_PLOT_DIR = r"E:\ML for prediction\plots_shallow_cnn_infer"
OUT_EXCEL_DIR = r"E:\ML for prediction\excel_shallow_cnn_infer"

MODEL_NAME = "best_shallow_cnn_regression"  # 输出 Excel：best_shallow_cnn_regression_0035.xlsx
N_KPOINTS = 31

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# 图像预处理：必须与训练一致（你训练时只做 Resize+ToTensor）
# ================================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


# ================================
# 读取 labels.csv（返回原始尺度 labels，不做归一化，便于直接对比）
# ================================
def load_labels_dict_original(csv_path):
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError("CSV 中必须包含 filename 列")

    label_cols = [c for c in df.columns if c != "filename"]
    label_df = df[label_cols].applymap(parse_complex_real).fillna(0.0)

    labels_orig = label_df.values.astype(np.float32)  # 原始物理尺度
    filenames = df["filename"].astype(str).values

    labels_dict = {f: v for f, v in zip(filenames, labels_orig)}
    return labels_dict, labels_orig.shape[1]


print("🔍 加载 labels.csv（原始尺度）...")
labels_dict, NUM_TARGETS = load_labels_dict_original(CSV_PATH)
print(f"✅ labels 加载完成：{len(labels_dict)} 条，维度 = {NUM_TARGETS}")


# ================================
# ✅ 与训练完全一致的浅层 CNN
# ================================
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


# ================================
# 加载模型 checkpoint
# ================================
print("🔍 加载 shallow CNN checkpoint ...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

num_targets_ckpt = checkpoint["num_targets"]
target_scale = float(checkpoint["target_scale"])  # 你的训练里是 1e5

if num_targets_ckpt != NUM_TARGETS:
    print(f"⚠ 警告：checkpoint num_targets={num_targets_ckpt}, labels 维度={NUM_TARGETS}")

model = ShallowCNNRegressor(num_targets=num_targets_ckpt)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(DEVICE).eval()

print("✅ 模型加载完成")
print(f"ℹ target_scale = {target_scale:g}")


# ================================
# R²（原始尺度）
# ================================
def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ================================
# 推理：输出预测能带 [n_bands, N_KPOINTS]（原始尺度）
# 训练时 label_norm = label / target_scale
# 所以推理输出 pred_norm -> pred = pred_norm * target_scale
# ================================
def predict_band_structure(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_norm = model(x).cpu().numpy().reshape(-1)  # [B] 归一化尺度

    pred = pred_norm * target_scale  # 转回原始尺度

    B = len(pred)
    if B % N_KPOINTS != 0:
        raise ValueError(f"预测维度 B={B} 不能被 N_KPOINTS={N_KPOINTS} 整除")

    n_bands = B // N_KPOINTS
    pred_bands = pred.reshape(N_KPOINTS, n_bands).T
    return pred_bands


# ================================
# 真实：输出真实能带 [n_bands, N_KPOINTS]（原始尺度）
# ================================
def get_true_band_structure(filename: str) -> np.ndarray:
    if filename not in labels_dict:
        raise KeyError(f"在 labels.csv 中找不到文件名: {filename}")

    vec = labels_dict[filename]
    B = len(vec)
    if B % N_KPOINTS != 0:
        raise ValueError(f"标签维度 B={B} 不能被 N_KPOINTS={N_KPOINTS} 整除")

    n_bands = B // N_KPOINTS
    true_bands = vec.reshape(N_KPOINTS, n_bands).T
    return true_bands


# ================================
# 画对比图：真实(实线) vs 预测(虚线)
# ================================
def plot_true_vs_pred(true_bands, pred_bands, filename, save_path):
    n_bands, n_k = true_bands.shape
    k = np.arange(n_k)

    plt.figure(figsize=(8, 6))
    for b in range(n_bands):
        plt.plot(k, true_bands[b], "-", alpha=0.85)
        plt.plot(k, pred_bands[b], "--", alpha=0.85)

    plt.xlabel("k-point index")
    plt.ylabel("Energy (original scale)")
    plt.title(f"True vs Predicted Band Structure\nSample: {filename}")

    true_line = plt.Line2D([], [], linestyle="-", color="black", label="True")
    pred_line = plt.Line2D([], [], linestyle="--", color="black", label="Pred")
    plt.legend(handles=[true_line, pred_line], loc="best")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 对比图已保存：{save_path}")


# ================================
# 导出单个 Excel（转置，便于 Origin）
# 文件命名：模型名_结构号.xlsx
# Sheet: true / pred
# 行：k 点；列：band_1..band_n；第一列：k_index
# ================================
def export_bands_to_single_excel_transposed(true_bands, pred_bands, model_name, structure_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    n_bands, n_k = true_bands.shape
    if pred_bands.shape != (n_bands, n_k):
        raise ValueError("pred_bands 与 true_bands 形状不一致")

    k_index = np.arange(n_k)
    band_cols = [f"band_{i+1}" for i in range(n_bands)]

    df_true = pd.DataFrame(true_bands.T, columns=band_cols)
    df_true.insert(0, "k_index", k_index)

    df_pred = pd.DataFrame(pred_bands.T, columns=band_cols)
    df_pred.insert(0, "k_index", k_index)

    out_path = os.path.join(out_dir, f"{model_name}_{structure_name}.xlsx")

    # 如果文件存在先删（确保没被 Excel/WPS 打开）
    if os.path.exists(out_path):
        os.remove(out_path)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_true.to_excel(writer, sheet_name="true", index=False)
        df_pred.to_excel(writer, sheet_name="pred", index=False)

    print(f"✅ Excel 已导出（适合 Origin）：{out_path}")


# ================================
# 主程序（单样本推理）
# ================================
if __name__ == "__main__":
    filename = "image_0035.png"  # TODO: 改成你想预测的那一张
    image_path = os.path.join(IMAGE_ROOT, filename)

    structure_name = os.path.splitext(filename)[0].split("_")[-1]

    print(f"📌 正在推理并对比：{image_path}")
    print(f"🏷 结构名：{structure_name}")

    true_bands = get_true_band_structure(filename)
    pred_bands = predict_band_structure(image_path)

    print("\n🎯 完成！")
    print("真实能带 shape:", true_bands.shape)
    print("预测能带 shape:", pred_bands.shape)

    r2 = r2_score_np(true_bands, pred_bands)
    print(f"📈 R² (orig scale, all bands & k-points): {r2:.6f}")

    os.makedirs(OUT_PLOT_DIR, exist_ok=True)
    plot_path = os.path.join(OUT_PLOT_DIR, f"band_compare_{os.path.splitext(filename)[0]}.png")
    plot_true_vs_pred(true_bands, pred_bands, filename, plot_path)

    export_bands_to_single_excel_transposed(
        true_bands=true_bands,
        pred_bands=pred_bands,
        model_name=MODEL_NAME,
        structure_name=structure_name,
        out_dir=OUT_EXCEL_DIR,
    )
