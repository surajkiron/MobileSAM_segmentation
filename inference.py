import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import MobileSAM, SegmentationDecoder
import torchvision.transforms.functional as F

def preprocess_image(np_image, image_shape):
  image_tensor = torch.from_numpy(np_image.transpose((2, 0, 1)))
  image_tensor = F.resize(image_tensor, image_shape)
  image_tensor = pad_image(image_tensor)
  return image_tensor

def pad_image(image_tensor):
  height = image_tensor.shape[-2]
  width = image_tensor.shape[-1]
  max_size = max(height, width)
  padding_left = (max_size - width) // 2
  padding_top = (max_size - height) // 2
  padding_right = max_size - width - padding_left
  padding_bottom = max_size - height - padding_top
  image_tensor = F.pad(image_tensor, [padding_left, padding_top, padding_right, padding_bottom])
  return image_tensor


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    return img
    
    
def main():    
    image = cv2.imread('notebooks/images/picture2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    encoder_weights = "weights/mobile_sam.pt"
    decoder_weights = "checkpoints/best_model.pth"
    model_type = "vit_t"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = MobileSAM(encoder_weights)
    decoder = SegmentationDecoder(decoder_weights)
    combined_model = nn.Sequential(encoder,decoder)
    combined_model.to(device=device)
    combined_model.eval()
    input = preprocess_image(image,image_shape)
    input = input.unsqueeze(dim=0).to(device=device,dtype=torch.float)/255.0
    masks = combined_model(input)
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    masks = show_anns(masks)
    plt.axis('off')
    plt.show() 
    cv2.imwrite("Mask1.jpg",masks)
    
    
if __name__ == '__main__':
    main()