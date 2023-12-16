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
  


  # -------------------------------------------------------------------------------------

from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from MobileSAM.mobile_sam.modeling import Sam, TwoWayTransformer, MaskDecoder, TinyViT, PromptEncoder
import torch

# ----------------------------------------------------------------------------

class MobileSAM(nn.Module):
  def __init__(self, path):
    super(MobileSAM, self).__init__()
    self.encoder = self.get_mobile_sam(path).image_encoder

  def forward(self, x):
    x = self.encoder(x)
    return x

  def get_mobile_sam(self, path):
    mobile_sam = Sam(
      image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8),
      prompt_encoder=PromptEncoder(
        embed_dim=256,
        image_embedding_size=(1024 // 16, 1024 // 16),
        input_image_size=(1024, 1024),
        mask_in_chans=16),
      mask_decoder=MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
          depth=2,
          embedding_dim=256,
          mlp_dim=2048,
          num_heads=8),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256),
      pixel_mean=[123.675, 116.28, 103.53],
      pixel_std=[58.395, 57.12, 57.375],
    )
    with open(path, "rb") as f:
        state_dict = torch.load(f)
    mobile_sam.load_state_dict(state_dict)
    return mobile_sam


# ----------------------------------------------------------------------------
