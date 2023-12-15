import warnings
import torch
import pytorch_lightning

from model import SegmentationDecoder
from loss import DiceLoss

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
  
class HumanDetectionNetwork(pytorch_lightning.LightningModule):
  def __init__(self, learning_rate):
    super().__init__()
    self.learning_rate = learning_rate
    self.decoder = SegmentationDecoder()
    self.loss_fn = DiceLoss()

  def forward(self, x):
    x = self.decoder(x)
    return x

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(
      params=self.parameters(),
      lr=self.learning_rate,
    )
    return {
      'optimizer': optimizer,
    }

  def training_step(self, train_batch):
    data, label = train_batch
    pred = self.forward(data)
    batch_loss = self.loss_fn(pred, label)
    self.log_dict({"train_loss": batch_loss}, prog_bar=True)
    return batch_loss
  
  def validation_step(self, valid_batch):
    data, label = valid_batch
    pred = self.forward(data)
    batch_loss = self.loss_fn(pred, label)
    self.log_dict({"valid_loss": batch_loss}, prog_bar=True)
    return batch_loss
  
  def test_step(self, test_batch):
    data, label = test_batch
    pred = self.forward(data)
    batch_loss = self.loss_fn(pred, label)
    self.log_dict({"test_loss": batch_loss}, prog_bar=True)
    return batch_loss
