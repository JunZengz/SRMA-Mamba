from scipy.spatial.distance import directed_hausdorff
import torch

def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

## https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/319452
def hd_dist(preds, targets):
    results = 0.0
    for i in range(preds.shape[0]):
        p, t = preds[i, :, :], targets[i, :, :]
        results += directed_hausdorff(p, t)[0]
    results = results/preds.shape[0]
    return results


def hd95_dist(preds, targets):
    results = 0.0
    for i in range(preds.shape[0]):
        p, t = preds[i ,: :, :, :], targets[i, : :, :, :]

        p_points = torch.nonzero(p).float()  # 预测值为 1 的点
        t_points = torch.nonzero(t).float()  # 目标值为 1 的点

        if p_points.shape[0] == 0 or t_points.shape[0] == 0:
            continue

        p_points_np = p_points.cpu().numpy()
        t_points_np = t_points.cpu().numpy()

        distances = directed_hausdorff(p_points_np, t_points_np)[0]
        hd95 = torch.quantile(distances, 0.95)
        results += hd95
    results = results / preds.shape[0]
    return results
