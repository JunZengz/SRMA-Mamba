import os
import time
from operator import add
import numpy as np
from tqdm import tqdm
import cv2
import nibabel as nib
import torch
from utils import create_dir, seeding, calculate_metrics2, calculate_metrics
from train import load_dataset
from metrics import dice_score
from monai import transforms
from monai.inferers import SlidingWindowInferer, SlidingWindowInfererAdapt
from configs.model_configs import build_SRMAMamba
from monai.inferers import sliding_window_inference
from torch.cuda.amp import autocast
import monai

def resize_and_save(img, path):
    resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(path, resized)

def print_score(metrics_score):
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    f2 = metrics_score[5]/len(test_x)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")


def print_score2(metrics_score):
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    f2 = metrics_score[5]/len(test_x)
    assd = metrics_score[6]/len(test_x)
    hd95 = metrics_score[7]/len(test_x)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - HD95: {hd95:.2f} - ASSD: {assd:.2f}")


def evaluate(model, save_path, test_x, test_y, size, transform, window_infer, device):
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        dir_name = x.split("/")[-1].split(".")[0]

        data = {"image":x, "label":y}
        augmented = transform(data)
        image_data = augmented["image"]
        label_data = augmented["label"]

        save_image = image_data
        image_data = torch.unsqueeze(image_data, dim=0)
        image_data = image_data.cuda()
        label_data = label_data.cuda()
        # print(image_data.shape, label_data.shape)

        """ Prediction """
        start_time = time.time()

        with torch.no_grad():
            y1, y2, y3, y4 = window_infer(image_data, model)
            pred = y1

            pred = torch.sigmoid(pred)
            pred = pred[0]

            end_time = time.time() - start_time
            time_taken.append(end_time)

        """ Evaluation Metrics """
        score = calculate_metrics2(label_data, pred)
        metrics_score = list(map(add, metrics_score, score))

        """ Saving Results """
        create_dir(os.path.join(save_path, "joint", dir_name))
        create_dir(os.path.join(save_path, "mask", dir_name))
        create_dir(os.path.join(save_path, "image", dir_name))
        create_dir(os.path.join(save_path, "gt", dir_name))

        save_image = nib.load(x).get_fdata()
        save_image = (save_image - save_image.min()) / (save_image.max() - save_image.min())
        save_label = label_data[0]

        pred = pred.detach().cpu().numpy()
        pred = (pred > 0.5).astype(np.uint8)
        save_pred = pred[0]

        for index in range(save_image.shape[-1]):
            image_slice = np.expand_dims(save_image[:,:,index], axis=-1) * 255
            label_slice = np.expand_dims(save_label[:,:,index], axis=-1) * 255
            pred_slice = np.expand_dims(save_pred[:,:, index], axis=-1) * 255

            # 转置，调整为 W × H 的顺序
            image_slice = np.transpose(image_slice, (1, 0, 2))  # W × H
            label_slice = np.transpose(label_slice, (1, 0, 2))  # W × H
            pred_slice = np.transpose(pred_slice, (1, 0, 2))  # W × H

            line = np.ones((save_image.shape[1], 10, 1)) * 255

            cat_image = np.concatenate([
                image_slice, line,
                label_slice, line,
                pred_slice
            ], axis=1)

            cv2.imwrite(os.path.join(save_path, "joint", dir_name, f"{index}.jpg"), cat_image)
            resize_and_save(image_slice, os.path.join(save_path, "image", dir_name, f"{index}.jpg"))
            resize_and_save(label_slice, os.path.join(save_path, "gt", dir_name, f"{index}.jpg"))
            resize_and_save(pred_slice, os.path.join(save_path, "mask", dir_name, f"{index}.jpg"))

        """ Save the results to 3D .nii.gz file """
        nifti_pred = nib.Nifti1Image(save_pred, affine=np.eye(4))
        nib.save(nifti_pred, os.path.join(save_path, "3d", f"{dir_name}.nii.gz"))

        nifti_image = nib.Nifti1Image((save_image).astype(np.uint8), affine=np.eye(4))
        nib.save(nifti_image, os.path.join(save_path, "3d_image", f"{dir_name}.nii.gz"))


    print_score2(metrics_score)
    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Choosing GPU """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """ Hyperparameters """
    size = [224, 224, 64] ## H, W, D
    batch_size = 1
    modality = 'T2'  # 'T1' Or 'T2'

    """" Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'SRMAMamba'
    model = build_SRMAMamba()

    model = model.to(device)

    save_path = os.path.join("results", f"files_{modality}", f'{model_name}')
    checkpoint_path = f"{save_path}/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Dataset """
    dataset_path = f"data/Cirrhosis_{modality}_3D"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")

    for item in ["mask", "joint", "3d", "image", "gt", "3d_image"]:
        create_dir(f"{save_path}/{item}")

    """ Transform """
    test_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstD(keys=["image", "label"], channel_dim="no_channel"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.ToTensord(keys=["image", "label"])
    ])

    """ Sliding Window Inference """
    window_infer = SlidingWindowInferer(roi_size=size, sw_batch_size=batch_size, overlap=0.25)

    create_dir(save_path)
    evaluate(model, save_path, test_x, test_y, size, test_transform, window_infer, device)
