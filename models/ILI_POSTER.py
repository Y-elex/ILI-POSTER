import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from .hyp_crossvit import *
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone
import cv2
import subprocess
import os
import sys
# 添加父目录到路径，以便导入 HVI_CIDNet
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from HVI_CIDNet.net.CIDNet import CIDNet
import json
import safetensors.torch as sf
from huggingface_hub import hf_hub_download
import argparse
import torchvision.transforms as transforms
import platform
from PIL import Image

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model


def is_low_light_image(
    img,
    mean_l_thresh=45,
    dark_ratio_thresh=0.55,
    contrast_thresh=25,
    entropy_thresh=6.5,
    sat_thresh=50
):
    # =============== 标准化输入为 [0,255] uint8 BGR ===============
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img[0]  # 取 batch 中第一张
        img = img.cpu().detach().numpy().transpose(1, 2, 0)  # [C,H,W] → [H,W,C]
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        # ⚠️ 明确假设输入为 RGB（PyTorch/PIL 标准），直接转 BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    elif isinstance(img, np.ndarray):
        # 处理 [0,1] float → [0,255] uint8
        if img.dtype in [np.float32, np.float64] and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        # ⚠️ 强制假设输入为 RGB（与数据加载流程一致）
        if img.shape[2] == 3:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            # 统一转为 BGR 供 OpenCV 使用
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Input must be 3-channel RGB image")
    else:
        raise TypeError("Input must be torch.Tensor or numpy.ndarray (RGB)")

    # =============== 低光判别逻辑（不变）===============
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    mean_l = L.mean()
    dark_ratio = np.mean(L < 40)
    contrast = L.std()

    hist = np.histogram(L, bins=256, range=(0, 255), density=True)[0]
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = hsv[:, :, 1].mean()

    score = 0
    score += mean_l < mean_l_thresh
    score += dark_ratio > dark_ratio_thresh
    score += contrast < contrast_thresh
    score += entropy < entropy_thresh
    score += sat_mean < sat_thresh

    is_low_light = score >= 3 and dark_ratio > dark_ratio_thresh

    metrics = {
        "mean_l": float(mean_l),
        "dark_ratio": float(dark_ratio),
        "contrast": float(contrast),
        "entropy": float(entropy),
        "sat_mean": float(sat_mean),
        "score": int(score)
    }
    return is_low_light, metrics


def load_cidnet_pretrained(model, pretrained_model_name_or_path: str):
    """
    加载 CIDNet 预训练权重（支持本地路径 / HuggingFace Hub）
    """
    model_id = str(pretrained_model_name_or_path)

    # 判断是否为本地路径
    # 首先尝试解析相对路径为绝对路径
    if not os.path.isabs(model_id):
        # 如果是相对路径，尝试基于当前文件所在目录解析
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        # 处理 ../ 开头的路径
        if model_id.startswith('../'):
            # 移除 ../ 前缀，然后拼接
            rel_path = model_id[3:]  # 移除 '../'
            abs_path = os.path.join(parent_dir, rel_path)
        elif model_id.startswith('./'):
            abs_path = os.path.join(parent_dir, model_id[2:])
        else:
            abs_path = os.path.join(parent_dir, model_id)
        
        # 检查绝对路径是否存在
        if os.path.exists(abs_path):
            model_id = abs_path
        elif os.path.exists(model_id):
            # 如果原始路径存在（相对于当前工作目录），使用它
            model_id = os.path.abspath(model_id)
    
    # 判断是否为本地路径（检查是否为 Hugging Face Hub 格式）
    # Hugging Face Hub 格式通常是 "username/repo_name" 且不包含路径分隔符（除了一个斜杠）
    is_hf_format = '/' in model_id and not os.path.isabs(model_id) and not os.path.exists(model_id) and not model_id.startswith('./') and not model_id.startswith('../') and model_id.count('/') == 1
    
    is_local_path = os.path.exists(model_id) or os.path.isabs(model_id)

    if is_local_path and not is_hf_format:
        if os.path.isdir(model_id):
            model_file = None
            for ext in [".safetensors", ".pth"]:
                candidate = os.path.join(model_id, f"model{ext}")
                if os.path.exists(candidate):
                    model_file = candidate
                    break
            if model_file is None:
                for f in os.listdir(model_id):
                    if f.endswith((".pth", ".safetensors")):
                        model_file = os.path.join(model_id, f)
                        break
            if model_file is None:
                raise FileNotFoundError(f"未在 {model_id} 中找到模型权重文件（.pth 或 .safetensors）")
        else:
            model_file = model_id
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"模型文件不存在: {model_file}")
    else:
        # Hugging Face Hub 模式
        try:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors",
                repo_type="model",
                local_files_only=True
            )
        except Exception:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    repo_type="model",
                    local_files_only=False
                )
            except Exception as e:
                raise FileNotFoundError(f"无法从 Hugging Face Hub 下载模型: {e}")

    if model_file.endswith(".safetensors"):
        state_dict = sf.load_file(model_file)
    else:
        state_dict = torch.load(model_file, map_location="cpu")

    model.load_state_dict(state_dict, strict=False)
    return model


def enhance_image_cidnet(
    img,
    model=None,
    model_path="../HVI_CIDNet/weights/LOLv1",
    alpha_s=1.0,
    alpha_i=1.0,
    gamma=1.0,
    device="cuda"
):
    """
    增强低光照图像
    支持 PIL.Image、图像路径或 PyTorch tensor 输入
    """
    # 处理不同类型的输入
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
        pil2tensor = transforms.ToTensor()
        input_tensor = pil2tensor(img)  # [3, H, W]
    elif isinstance(img, Image.Image):
        pil2tensor = transforms.ToTensor()
        input_tensor = pil2tensor(img)  # [3, H, W]
    elif isinstance(img, torch.Tensor):
        # 已经是 tensor，假设形状为 [B, C, H, W] 或 [C, H, W]
        if img.dim() == 4:
            input_tensor = img[0]  # 取第一个图像 [C, H, W]
        else:
            input_tensor = img  # [C, H, W]
        # 确保值在 [0, 1] 范围
        if input_tensor.max() > 1.0:
            input_tensor = input_tensor / 255.0
    else:
        raise TypeError("img 必须是 PIL.Image、图像路径或 PyTorch tensor")

    # 确保 tensor 在正确的设备上
    if input_tensor.device != device:
        input_tensor = input_tensor.to(device)

    # ---------- 2. padding ----------
    factor = 8
    h, w = input_tensor.shape[1:]
    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0

    input_tensor = F.pad(
        input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor,
        (0, padw, 0, padh),
        mode="reflect"
    )

    # ---------- 3. 加载或使用传入的模型 ----------
    if model is None:
        model = CIDNet().to(device)
        model = load_cidnet_pretrained(model, model_path)
        model.eval()

    model.trans.alpha_s = alpha_s
    model.trans.alpha = alpha_i
    model.trans.gated = True
    model.trans.gated2 = True

    # ---------- 4. 推理 ----------
    with torch.no_grad():
        output = model((input_tensor.to(device)) ** gamma)
        output = torch.clamp(output, 0, 1)
        output = output[:, :, :h, :w]

    # 返回 tensor 而不是 PIL Image，以便在训练流程中使用
    return output


class LowLightEnhancer(nn.Module):
    def __init__(self, prob=1.0, model_path="../HVI_CIDNet/weights/LOLv1", device="cuda",thresholds=None):
        super().__init__()
        self.thresholds = thresholds or {
            "mean_l": 45, "dark_ratio": 0.55, "contrast": 25,
            "entropy": 6.5, "sat": 50, "min_score": 3
        }
        self.prob = prob
        self.device = device
        self.model_path = model_path
        self.cidnet = CIDNet().to(device)
        self.cidnet = load_cidnet_pretrained(self.cidnet, model_path)
        self.cidnet.eval()
        for param in self.cidnet.parameters():
            param.requires_grad = False

    def _load_cidnet(self):
        pass

    def is_low_light_batch(self, img_batch):
        """
        img_batch: [B, C, H, W] in [0, 1], on GPU
        Returns: bool tensor [B]
        """
        B = img_batch.shape[0]
        is_low_list = []
        for i in range(B):
            is_low, _ = is_low_light_image(img_batch[i].detach())  # is_low_light_image handles tensor
            is_low_list.append(is_low)
        return torch.tensor(is_low_list, device=img_batch.device)

    def enhance_batch(self, img_batch, low_mask):
        """
        Enhance only images where low_mask is True.
        img_batch: [B, C, H, W] in [0,1]
        low_mask: [B] bool
        """
        if not low_mask.any():
            return img_batch

        self._load_cidnet()
        enhanced = img_batch.clone()

        # Process only low-light images
        low_indices = torch.where(low_mask)[0]
        for idx in low_indices:
            single_img = img_batch[idx:idx+1]  # [1, C, H, W]
            with torch.no_grad():
                # enhance_image_cidnet modified to accept batch tensor and return same
                enhanced_img = self._enhance_single(single_img)
                enhanced[idx] = enhanced_img[0]

        return enhanced

    def _enhance_single(self, img_tensor):
        """img_tensor: [1, C, H, W] in [0,1]"""
        factor = 8
        _, _, h, w = img_tensor.shape
        H = ((h + factor) // factor) * factor
        W = ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0

        padded = F.pad(img_tensor, (0, padw, 0, padh), mode="reflect")
        gamma = 1.0
        alpha_s = 1.0
        alpha_i = 1.0

        self.cidnet.trans.alpha_s = alpha_s
        self.cidnet.trans.alpha = alpha_i
        self.cidnet.trans.gated = True
        self.cidnet.trans.gated2 = True

        with torch.no_grad():
            output = self.cidnet((padded) ** gamma)
            output = torch.clamp(output, 0, 1)
            output = output[:, :, :h, :w]
        return output

    def forward(self, x):
        """
        x: [B, C, H, W] in [0, 1], on GPU
        Returns: enhanced x in [0, 1]
        """
        if self.prob <= 0:
            return x

        # Deterministic during validation; stochastic during training
        if self.training and torch.rand(1).item() >= self.prob:
            return x

        low_mask = self.is_low_light_batch(x)
        x_enhanced = self.enhance_batch(x, low_mask)
        return x_enhanced


class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat


class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=7, type="large", negative_emotions=None, neg_weight=2.0):
        super().__init__()
        depth = 8
        if type == "small": depth = 4
        if type == "base": depth = 6

        self.img_size = img_size
        self.num_classes = num_classes

        if negative_emotions is None:
            # 默认 RAF-DB 负面情绪索引
            negative_emotions = [2, 4, 5, 6]  # sadness, fear, disgust, anger
        
        # 构建类别权重向量 [1, 1, ..., w, ..., 1]
        class_weights = torch.ones(num_classes)
        class_weights[negative_emotions] = neg_weight
        self.register_buffer('class_weights', class_weights)  # 不参与训练，但可随模型保存


        # Low-light enhancement module (inside model)
        # ⚠️ 注意：lowlight_enhancer 要求输入为 [0,1] RGB（ToTensor 后，未 Normalize）
        self.lowlight_enhancer = LowLightEnhancer(prob=1.0, model_path="../HVI_CIDNet/weights/LOLv1")

        # Face landmark backbone
        self.face_landback = MobileFaceNet([112, 112], 136)
        face_ckpt = torch.load('./models/pretrain/mobilefacenet_model_best.pth.tar', map_location='cpu')
        self.face_landback.load_state_dict(face_ckpt['state_dict'])
        self.face_landback.eval()
        for p in self.face_landback.parameters():
            p.requires_grad = False

        # IR backbone
        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_ckpt = torch.load('./models/pretrain/ir50.pth', map_location='cpu')
        self.ir_back = load_pretrained_weights(self.ir_back, ir_ckpt)
        self.ir_layer = nn.Linear(1024, 512)

        # Fusion
        self.pyramid_fuse = HyVisionTransformer(
            in_chans=49, q_chanel=49, embed_dim=512,
            depth=depth, num_heads=8, mlp_ratio=2.,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1
        )
        self.se_block = SE_block(input_dim=512)
        self.head = ClassificationHead(input_dim=512, target_dim=num_classes)

        # ImageNet normalization (applied AFTER enhancement)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """
        x: [B, C, H, W] in [0, 1] (from ToTensor, no Normalize)
        """
        # Step 1: Low-light enhancement (if needed)
        x = self.lowlight_enhancer(x)  # still in [0, 1]

        # Step 2: Apply ImageNet normalization (critical for pretrained backbones)
        x_norm = (x - self.mean) / self.std

        # Step 3: Feature extraction
        B_ = x_norm.shape[0]
        x_face = F.interpolate(x_norm, size=112, mode='bilinear', align_corners=False)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1, 2)  # [B, 49, 512]

        x_ir = self.ir_back(x_norm)
        x_ir = self.ir_layer(x_ir)  # [B, 512] → but expected [B, 49, 512]?

        # ⚠️ 注意：这里可能有维度不匹配！
        # 如果 ir_back 输出是 [B, 1024]，经过 Linear(1024→512) 后是 [B, 512]
        # 但 pyramid_fuse 期望 [B, 49, 512]
        # 你需要 reshape 或 repeat：
        if x_ir.dim() == 2:
            x_ir = x_ir.unsqueeze(1).expand(-1, 49, -1)  # [B, 512] → [B, 49, 512]

        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        out = self.head(y_hat)

        return out, y_hat