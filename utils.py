import os
import numpy as np
import torch
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from metrics import precision, recall, F2, dice_score, jac_score
from monai.metrics import SurfaceDistanceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot

""" Seeding the randomness. """
def seeding(seed):
    # random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Shuffle the dataset. """
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

def calculate_metrics(y_true, y_pred):
    ## Tensor processing
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)

    ## Reshape
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]



def calculate_metrics2(y_true, y_pred):
    y_pred_origin = (y_pred > 0.5).unsqueeze(0) if y_pred.dim() == 4 else (y_pred > 0.5)
    y_true_origin = (y_true > 0.5).unsqueeze(0) if y_true.dim() == 4 else (y_true > 0.5)
    ## Tensor processing
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)

    ## Reshape
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    y_pred_onehot = one_hot(y_pred_origin, num_classes=2)
    y_true_onehot = one_hot(y_true_origin, num_classes=2)
    score_assd = SurfaceDistanceMetric(include_background=False, symmetric=True)(y_pred_onehot, y_true_onehot).cpu().detach().numpy().item()

    score_hd95 = HausdorffDistanceMetric(include_background=False, percentile=95)(y_pred_onehot, y_true_onehot).cpu().detach().numpy().item()
    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta, score_assd, score_hd95]