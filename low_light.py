import os
import cv2
import numpy as np

from lowlight_create import synthesize_real_lowlight

def synthesize_lowlight_clean(img):
    img = img.astype(np.float32) / 255.0

    # ① 曝光衰减（核心）
    gamma = np.random.uniform(2.4, 3.2)
    beta = np.random.uniform(0.25, 0.45)
    low = beta * np.power(img, gamma)

    # ② 对比度压缩（低光关键特征）
    mean = np.mean(low, axis=(0,1), keepdims=True)
    low = (low - mean) * 0.55 + mean

    # ③ 轻微色偏（夜间常见）
    color_shift = np.array([
        np.random.uniform(0.85, 1.0),  # R
        np.random.uniform(0.95, 1.1),  # G
        np.random.uniform(0.75, 0.9)   # B
    ])
    low = low * color_shift

    # ④ 暗部进一步压塌
    low = np.clip(low, 0, 1)
    low = low ** 1.2

    return (low * 255).astype(np.uint8)


src_dir = r"E:\python_code\FER\datasets\before_RAF-DB\train\6"
dst_dir = r"E:\python_code\FER\datasets\lowlight_before_RAF-DB\train\6"

os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    img_path = os.path.join(src_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告: 无法读取 {filename}，已跳过")
        continue

    low_img = synthesize_lowlight_clean(img)

    save_path = os.path.join(dst_dir, filename)  # ⭐关键
    cv2.imwrite(save_path, low_img)