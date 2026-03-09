import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ================================
# 从训练脚本导入
# ================================
from train_swin import (
    create_swin_model,
    TARGET_SCALE,   # 这里不一定用到，但保留导入不影响
    CSV_PATH,
    parse_complex_real,
)

# ================================
# 基本参数
# ================================
N_KPOINTS = 31  # 每条结构有多少个 k 点

IMAGE_ROOT = r"E:\ML for prediction\data\train\images"
CHECKPOINT_PATH = r"E:\ML for prediction\model\best_swin_regression.pth"

OUT_PLOT_DIR = r"E:\ML for prediction\plots_swin_infer"
OUT_EXCEL_DIR = r"E:\ML for prediction\excel_swin_infer"

MODEL_NAME = "best_swin_regression"  # 输出 Excel 文件名：best_swin_regression_0035.xlsx

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# 图像预处理（必须与训练一致）
# ================================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ================================
# 读取 labels.csv
# ================================
def load_labels_dict(csv_path):
    """
    返回：
      labels_dict: {filename(str): np.ndarray([B], float32)}
      B: 标签维度
    """
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError("CSV 中必须包含 filename 列")

    label_cols = [c for c in df.columns if c != "filename"]

    # 解析复数为实数并填空
    label_df = df[label_cols].applymap(parse_complex_real).fillna(0.0)

    labels = label_df.values.astype(np.float32)  # [N, B]
    filenames = df["filename"].astype(str).values

    labels_dict = {f: v for f, v in zip(filenames, labels)}
    return labels_dict, labels.shape[1]


print("🔍 加载 labels.csv ...")
labels_dict, NUM_TARGETS = load_labels_dict(CSV_PATH)
print(f"✅ labels 加载完成：{len(labels_dict)} 条，维度 = {NUM_TARGETS}")

# ================================
# 加载模型
# ================================
print("🔍 加载模型 ...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

num_targets_ckpt = checkpoint["num_targets"]
target_scale = checkpoint["target_scale"]

if num_targets_ckpt != NUM_TARGETS:
    print(f"⚠ 警告：checkpoint num_targets={num_targets_ckpt}, labels 维度={NUM_TARGETS}")

model = create_swin_model(num_targets_ckpt)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("✅ 模型加载完成")

# ================================
# 计算 R²（决定系数）
# ================================
def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    y_true, y_pred: 任意shape，内部会展平成一维
    返回 R^2 = 1 - SSE/SST
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # 避免 y_true 全常数导致除零
    if ss_tot == 0:
        return float("nan")

    return 1.0 - (ss_res / ss_tot)

# ================================
# 推理：输出预测能带 [n_bands, N_KPOINTS]
# ================================
def predict_band_structure(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_norm = model(x).cpu().numpy().reshape(-1)  # [B] 归一化预测

    pred = pred_norm * target_scale  # 反归一化到物理尺度

    B = len(pred)
    if B % N_KPOINTS != 0:
        raise ValueError(f"预测维度 B={B} 不能被 N_KPOINTS={N_KPOINTS} 整除")

    n_bands = B // N_KPOINTS
    pred_bands = pred.reshape(N_KPOINTS, n_bands).T  # [n_bands, N_KPOINTS]
    return pred_bands


# ================================
# 真实：输出真实能带 [n_bands, N_KPOINTS]
# ================================
def get_true_band_structure(filename):
    if filename not in labels_dict:
        raise KeyError(f"在 labels.csv 中找不到文件名: {filename}")

    vec = labels_dict[filename]  # [B]

    B = len(vec)
    if B % N_KPOINTS != 0:
        raise ValueError(f"标签维度 B={B} 不能被 N_KPOINTS={N_KPOINTS} 整除")

    n_bands = B // N_KPOINTS
    true_bands = vec.reshape(N_KPOINTS, n_bands).T  # [n_bands, N_KPOINTS]
    return true_bands


# ================================
# 画对比图：真实(实线) vs 预测(虚线)
# ================================
def plot_true_vs_pred(true_bands, pred_bands, filename, save_path):
    n_bands, n_k = true_bands.shape
    k = np.arange(n_k)

    plt.figure(figsize=(8, 6))
    for b in range(n_bands):
        plt.plot(k, true_bands[b], "-", alpha=0.8)
        plt.plot(k, pred_bands[b], "--", alpha=0.8)

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
# 导出单个 Excel（转置，便于粘贴到 Origin）
# 文件命名：模型名_结构名.xlsx
# 结构名从 filename 中解析，如 image_0035.png -> 0035
# 结构：
#   行：k 点
#   列：band_1, band_2, ...
#   第一列：k_index
#   Sheet: true / pred
# ================================
def export_bands_to_single_excel_transposed(true_bands, pred_bands, model_name, structure_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    n_bands, n_k = true_bands.shape
    if pred_bands.shape != (n_bands, n_k):
        raise ValueError(f"pred_bands shape {pred_bands.shape} 与 true_bands shape {true_bands.shape} 不一致")

    k_index = np.arange(n_k)
    band_cols = [f"band_{i+1}" for i in range(n_bands)]

    # 转置： [n_k, n_bands]
    df_true = pd.DataFrame(true_bands.T, columns=band_cols)
    df_true.insert(0, "k_index", k_index)

    df_pred = pd.DataFrame(pred_bands.T, columns=band_cols)
    df_pred.insert(0, "k_index", k_index)

    out_path = os.path.join(out_dir, f"{model_name}_{structure_name}.xlsx")

    # 防止 PermissionError：若文件存在先删除（前提：Excel/WPS 没打开该文件）
    if os.path.exists(out_path):
        os.remove(out_path)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_true.to_excel(writer, sheet_name="true", index=False)
        df_pred.to_excel(writer, sheet_name="pred", index=False)

    print(f"✅ 转置后的 Excel 已导出（适合 Origin）：{out_path}")


# ================================
# 主程序
# ================================
if __name__ == "__main__":
    # 你想测试的图片文件名（必须和 labels.csv 里的 filename 完全一致）
    filename = "image_3336.png"  # TODO: 改成你想看的那一张
    image_path = os.path.join(IMAGE_ROOT, filename)

    # 从文件名解析结构名：image_0035.png -> 0035
    structure_name = os.path.splitext(filename)[0].split("_")[-1]

    print(f"📌 正在推理并对比：{image_path}")
    print(f"🏷 结构名：{structure_name}")

    # 真实能带
    true_bands = get_true_band_structure(filename)
    # 预测能带
    pred_bands = predict_band_structure(image_path)

    print("\n🎯 完成！")
    print("真实能带 shape:", true_bands.shape)
    print("预测能带 shape:", pred_bands.shape)

    # ✅ 计算并打印 R²（整体：所有 band × 所有 k 点）
    r2 = r2_score_np(true_bands, pred_bands)
    print(f"📈 R² (all bands & k-points): {r2:.6f}")

    # 画对比图
    os.makedirs(OUT_PLOT_DIR, exist_ok=True)
    plot_path = os.path.join(OUT_PLOT_DIR, f"band_compare_{os.path.splitext(filename)[0]}.png")
    plot_true_vs_pred(true_bands, pred_bands, filename, plot_path)

    # 导出 Excel（只生成一个文件，且转置；命名含结构号）
    export_bands_to_single_excel_transposed(
        true_bands=true_bands,
        pred_bands=pred_bands,
        model_name=MODEL_NAME,
        structure_name=structure_name,
        out_dir=OUT_EXCEL_DIR,
    )
