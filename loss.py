import torch.nn as nn
import monai

#----------------------------------------------------------------------------

class DiceLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
  def forward(self, pred, label):
    pred = pred.view(-1, pred.shape[-2], pred.shape[-1])
    label = label.view(-1, label.shape[-2], label.shape[-1])
    return self.loss_fn(pred, label)
  