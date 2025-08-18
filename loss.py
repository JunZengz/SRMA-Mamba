import torch
import monai
import torch.nn as nn
from monai import losses

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predictions, targets):
        loss = self._loss(predictions, targets)
        return loss

class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions, tragets):
        loss = self._loss(predictions, tragets)
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True
        )

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss

class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceCELoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True
        )

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss
