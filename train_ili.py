import warnings

warnings.filterwarnings("ignore")
# from apex import amp
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import os
import pandas as pd
import cv2
import torchvision.transforms as transforms
from PIL import Image
import safetensors.torch as sf
from huggingface_hub import hf_hub_download
import torch
import argparse
from HVI_CIDNet.net.CIDNet import CIDNet
import torch.nn.functional as F
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class

from sklearn.metrics import f1_score, confusion_matrix
from time import time
import matplotlib.pyplot as plt
from utils import *
from data_preprocessing.sam import SAM
from models.ILI_POSTER import pyramid_trans_expr
from torchvision.datasets import ImageFolder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FERPlus', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.000004, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200, help='Total training epochs.')
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    return parser.parse_args()


def get_imagefolder_dataset(dataset_name, split, transform):
    root = os.path.join('../', 'datasets', dataset_name, split)
    return ImageFolder(root=root, transform=transform)

def run_training():
    args = parse_args()
    torch.manual_seed(123)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    negative_emotions = [2, 4, 5, 6]


    data_transforms = transforms.Compose([
        #transforms.ToPILImage(),数据集为rafdb时再加上
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02, 0.1)),
    ])

    data_transforms_val = transforms.Compose([
        #transforms.ToPILImage()，数据集为rafdb时再加上
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    num_classes = 7

    if args.dataset == "affectnet_folder" or args.dataset == "ferplus":
        num_classes = 8
        # AffectNet 8-class: 0: neutral, 1: happiness, 2: sadness, 3: surprise, 4: fear, 5: disgust, 6: anger, 7: contempt
        negative_emotions = [2, 4, 5, 6, 7]  # sadness, fear, disgust, anger, contempt

    # 构建权重向量：负面情绪权重设为 2.0，其余为 1.0
    class_weights = torch.ones(num_classes)
    class_weights[negative_emotions] = 2.0  # 可调整为 1.5~3.0 实验
    class_weights = class_weights.cuda()  # 与模型同设备
    if args.dataset == "rafdb":
        datapath = './data/raf-basic/'
        num_classes = 7
        train_dataset = RafDataSet(datapath, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = RafDataSet(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet":
        datapath = './data/AffectNet/'
        num_classes = 7
        train_dataset = Affectdataset(datapath, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = Affectdataset(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8class":
        datapath = './data/AffectNet/'
        num_classes = 8
        train_dataset = Affectdataset_8class(datapath, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = Affectdataset_8class(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)


    elif args.dataset == 'merge_dataset':
        datapath = './merge_dataset/'
        data_val = './data/raf-basic/'
        num_classes = 8
        train_dataset = get_imagefolder_dataset('merge_dataset', "train", data_transforms)
        val_dataset = RafDataSet(data_val, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset.lower() in ["lowlight_RAF-DB", "raf-db", "affectnet_folder", "fer-2013", "ferplus","before_raf-db"]:
        # 新增支持ImageFolder结构的数据集
        dataset_map = {
            "lowlight_raf-db": "lowlight_RAF-DB",
            "raf-db": "RAF-DB",
            "affectnet_folder": "AffectNet",
            "fer-2013": "FER-2013",
            "ferplus": "FERPlus",
            "before_raf-db": "before_RAF-DB"
        }
        dataset_dir = dataset_map[args.dataset.lower()]
        train_dataset = get_imagefolder_dataset(dataset_dir, "train", data_transforms)
        val_dataset = get_imagefolder_dataset(dataset_dir, "val", data_transforms_val)
        # 自动推断类别数
        num_classes = len(train_dataset.classes)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype, negative_emotions=negative_emotions)
        #model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    else:
        return print('dataset name is not correct')

    val_num = val_dataset.__len__()
    print('Train set size:', train_dataset.__len__())
    print('Validation set size:', val_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               # sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True)


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False)

    # model = Networks.ResNet18_ARM___RAF()

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    print("batch_size:", args.batch_size)

    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        checkpoint = checkpoint["model_state_dict"]
        model = load_pretrained_weights(model, checkpoint)

    params = model.parameters()
    if args.optimizer == 'adamw':
        # base_optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        # base_optimizer = torch.optim.Adam(params, args.lr, weight_decay=1e-4)
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        # base_optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=1e-4)
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")
    # print(optimizer)
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False,)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)
    CE_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    lsce_criterion = LabelSmoothingCrossEntropy(weight=class_weights, smoothing=0.2)


    logs = []
    best_acc = 0
    for i in range(1, args.epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        start_time = time()
        model.train()
        for batch_i, (imgs, targets) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            outputs, features = model(imgs)
            targets = targets.cuda()

            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            loss = 2 * lsce_loss + CE_loss
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            outputs, features = model(imgs)
            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)

            loss = 2 * lsce_loss + CE_loss
            loss.backward() # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)


            train_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

            # 在 train_loader 循环内（例如每 50 个 batch）
            if batch_i % 50 == 0:
                with torch.no_grad():
                    # 注意：model 是 DataParallel，需通过 .module 访问原始模型
                    low_mask = model.module.lowlight_enhancer.is_low_light_batch(imgs)
                    low_ratio = low_mask.float().mean().item()
                    print(f"[Batch {batch_i}] Low-light ratio: {low_ratio:.1%} "
                        f"(e.g., {low_mask[:5].cpu().numpy()} for first 5 imgs)")

        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_loss = train_loss / iter_cnt
        elapsed = (time() - start_time) / 60

        print('[Epoch %d] Train time:%.2f, Training accuracy:%.4f. Loss: %.3f LR:%.6f' %
              (i, elapsed, train_acc, train_loss, optimizer.param_groups[0]["lr"]))

        scheduler.step()

        pre_labels = []
        gt_labels = []
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            for batch_i, (imgs, targets) in enumerate(val_loader):
                outputs, features = model(imgs.cuda())
                targets = targets.cuda()

                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss

                val_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()

            val_loss = val_loss / iter_cnt
            val_acc = bingo_cnt.float() / float(val_num)
            val_acc = np.around(val_acc.numpy(), 4)
            f1 = f1_score(pre_labels, gt_labels, average='macro')
            total_socre = 0.67 * f1 + 0.33 * val_acc

            print("[Epoch %d] Validation accuracy:%.4f, Loss:%.3f, f1 %4f, score %4f" % (
            i, val_acc, val_loss, f1, total_socre))


            if val_acc > best_acc:
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('./checkpoint', "epoch" + str(i) + "_acc_ili_negative_enhance" + str(val_acc) + args.dataset + ".pth"))
                print('Model saved.')
            if val_acc > best_acc:
                best_acc = val_acc
                print("best_acc:" + str(best_acc))
        logs.append([i, float(train_acc), float(train_loss), float(val_acc), float(val_loss)])

    df = pd.DataFrame(logs, columns=['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss'])
    os.makedirs('./logs', exist_ok=True)
    df.to_excel(f'./logs/train_log_ili_{args.dataset}_negative_enhance.xlsx', index=False)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
    plt.plot(df['epoch'], df['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)  # 确保目录存在
    plt.savefig(f'figures/POSTER_accuracy_ili_negative_enhance_loss_plot_{args.dataset}.png')


if __name__ == "__main__":
    run_training()