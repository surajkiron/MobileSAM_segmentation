import torch.nn as nn
from torch.nn import functional as F

class SegmentationDecoder(nn.Module):
  def __init__(self):
    super(SegmentationDecoder, self).__init__()
    self.decoder = nn.Sequential(*[nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                                   nn.GELU(),
                                   nn.UpsamplingBilinear2d(scale_factor=2),
                                   nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                                   nn.GELU(),
                                   nn.UpsamplingBilinear2d(scale_factor=2),
                                   nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                                   nn.UpsamplingBilinear2d(scale_factor=4)])

  def forward(self, x):
    x = self.decoder(x)
    return x