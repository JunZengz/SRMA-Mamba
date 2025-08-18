import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import random
import datetime, time
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import seeding, create_dir, print_and_save, epoch_time, calculate_metrics
from monai import transforms
from loss import DiceCELoss
from monai.inferers import SlidingWindowInferer
from configs.model_configs import build_SRMAMamba
import monai
from torch.cuda.amp import GradScaler, autocast

def load_dataset(path):
    train_x = sorted(glob(os.path.join(path, "train_images", "*.nii.gz")))
    train_y = sorted(glob(os.path.join(path, "train_masks", "*.nii.gz")))

    valid_x = sorted(glob(os.path.join(path, "valid_images", "*.nii.gz")))
    valid_y = sorted(glob(os.path.join(path, "valid_masks", "*.nii.gz")))

    test_x = sorted(glob(os.path.join(path, "test_images", "*.nii.gz")))
    test_y = sorted(glob(os.path.join(path, "test_masks", "*.nii.gz")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


class DATASET(Dataset):
    def __init__(self, images, labels, size, transform=None):
        super().__init__()

        self.images = images
        self.labels = labels
        self.size = size
        self.transform = transform
        self.n_samples = len(images)

    def __getitem__(self, index):
        ## Channel, Depth, Height, Width
        image_path = self.images[index]
        label_path = self.labels[index]

        """ Transforms """
        if self.transform is not None:
            data = {
                "image": image_path,
                "label": label_path
            }

            augmented = self.transform(data)
            image_data = augmented["image"]
            label_data = augmented["label"]

        return image_data, label_data

    def __len__(self):
        return self.n_samples

def valid_collate_fn(batch):
    return batch

def train(model, loader, optimizer, loss_fn, epoch, scaler, device):
    model.train()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    skip_count = 0
    progress_bar = tqdm(loader, desc="Training Epoch {}".format(epoch + 1), total=len(loader))
    for i, (image, label) in enumerate(progress_bar):
        image = image.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        with autocast():  #
            y1, y2, y3, y4 = model(image)
            y_pred = y1
            loss1 = loss_fn(y1, label)
            loss2 = loss_fn(y2, label)
            loss3 = loss_fn(y3, label)
            loss4 = loss_fn(y4, label)
            loss = loss1 + loss2 + loss3 + loss4

        # 防止 NaN 或 Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Loss is NaN or Inf at batch {i}, skipping...")
            skip_count += 1
            continue  # 跳过该批次

        scaler.scale(loss).backward()
        # 加梯度裁剪
        scaler.unscale_(optimizer)  # 先unscale
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        progress_bar.set_postfix(loss=loss.item(), refresh=True)

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        y_pred = torch.sigmoid(y_pred)
        for yt, yp in zip(label, y_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_loss += loss.item()
        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)


    valid_batch_count = len(loader) - skip_count
    epoch_loss = epoch_loss / valid_batch_count
    epoch_jac = epoch_jac / valid_batch_count
    epoch_f1 = epoch_f1 / valid_batch_count
    epoch_recall = epoch_recall / valid_batch_count
    epoch_precision = epoch_precision / valid_batch_count

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

def evaluate(window_infer, model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0


    with torch.no_grad():
        for i, data in enumerate(loader):
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []
            for (image, label) in data:
                image = image.unsqueeze(0).to(device, dtype=torch.float32)
                label = label.unsqueeze(0).to(device, dtype=torch.float32)

                with autocast():
                    y1, y2, y3, y4 = window_infer(image, model)
                    y_pred = y1
                    loss1 = loss_fn(y1, label)
                    loss2 = loss_fn(y2, label)
                    loss3 = loss_fn(y3, label)
                    loss4 = loss_fn(y4, label)
                    loss = loss1 + loss2 + loss3 + loss4

                epoch_loss += loss.item()

                """ Calculate the metrics """
                y_pred = torch.sigmoid(y_pred)
                score = calculate_metrics(label, y_pred)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            """ Calculate the metrics """

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss / len(loader)
        epoch_jac = epoch_jac / len(loader)
        epoch_f1 = epoch_f1 / len(loader)
        epoch_recall = epoch_recall / len(loader)
        epoch_precision = epoch_precision / len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Hyperparameters """
    size = [224, 224, 64] ## H, W, D
    batch_size = 2
    num_epochs = 500
    lr = 1e-4
    num_classes = 1
    early_stopping_patience = 50
    modality = 'T2'  # 'T1' Or 'T2'

    """ Model """
    device = torch.device("cuda")

    model_name = 'SRMAMamba'
    model = build_SRMAMamba()

    model = model.to(device)

    """ Directories """
    save_path = os.path.join("results", f"files_{modality}", f"{model_name}")
    os.makedirs(save_path, exist_ok=True)

    checkpoint_path = f"{save_path}/checkpoint.pth"
    dataset_path = f"data/Cirrhosis_{modality}_3D"

    # create_dir("files")

    """ Training logfile """
    train_log_path = f"{save_path}/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open(train_log_path, "w")
        train_log.write("\n")
        train_log.close()

    print_and_save(train_log_path, dataset_path)

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")

    """ Transforms """
    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstD(keys=["image", "label"], channel_dim="no_channel"),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=size),
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=size, random_size=False),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        transforms.ToTensord(keys=["image", "label"])
    ])
    valid_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstD(keys=["image", "label"], channel_dim="no_channel"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.ToTensord(keys=["image", "label"])
    ])

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, transform=train_transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=valid_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=valid_collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, verbose=5)
    loss_fn = DiceCELoss()
    loss_name = "DiceCELoss"

    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Sliding Window Inference """
    window_infer = SlidingWindowInferer(roi_size=size, sw_batch_size=batch_size, overlap=0.25)

    """ Training the SRMAMamba """
    best_valid_metrics = 0.0
    early_stopping_count = 0
    scaler = GradScaler()

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, epoch, scaler, device)
        valid_loss, valid_metrics = evaluate(window_infer, model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
