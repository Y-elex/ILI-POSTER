import torch.utils.data as data
import cv2
import pandas as pd
import os
# import image_utils
import random
import cv2
import numpy as np



import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class RafDataSet(Dataset):
    def __init__(self, raf_path, train=True, transform=None, basic_aug=False):
        self.raf_path = raf_path
        self.transform = transform
        self.train = train
        self.basic_aug = basic_aug

        # Load correct CSV
        label_file = 'train_labels.csv' if train else 'test_labels.csv'
        label_path = os.path.join(self.raf_path, 'EmoLabel', label_file)
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        df = pd.read_csv(label_path)
        assert 'image' in df.columns and 'label' in df.columns, \
            f"CSV must contain 'image' and 'label' columns. Found: {list(df.columns)}"

        self.file_names = df['image'].values
        labels = df['label'].values
        # 如果标签是1-based（最小值为1），则转换为0-based；如果已经是0-based（最小值为0），则保持不变
        if labels.min() > 0:
            self.labels = labels - 1  # Convert 1–7 → 0–6 for PyTorch
        else:
            self.labels = labels  # Already 0-based (0–6)

        print(f"✅ Loaded {len(self)} samples from {label_file} ({'train' if train else 'test'})")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        label = int(self.labels[idx])

        # --- Try aligned first, fallback to root Image/ ---
        aligned_path = os.path.join(self.raf_path, 'Image', 'aligned', img_name)
        basic_path = os.path.join(self.raf_path, 'Image', img_name)

        if os.path.exists(aligned_path):
            img_path = aligned_path
        elif os.path.exists(basic_path):
            img_path = basic_path
        else:
            raise FileNotFoundError(
                f"Image not found. Tried:\n  1. {aligned_path}\n  2. {basic_path}"
            )

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)