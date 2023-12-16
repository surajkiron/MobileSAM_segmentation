import h5py
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from pycocotools.coco import COCO
import os
import torchvision.transforms.functional as F
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from model import MobileSAM
from util import ResizeLongestSide
from config import parse_args, save_args



def get_metadata(coco):
  cat_ids = coco.getCatIds(catNms=['person'])
  ids = coco.getImgIds(catIds=cat_ids)[0:100]              # first 100 person images
  for img_id in ids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)                         # list[segmentation_mask, area, iscrowd(bool), image_id, bbox, category_id, id]
    if len(anns) >= 1 and anns[0]['segmentation'] and anns[0]['category_id']==cat_ids[0]:
      image_name = coco.imgs[img_id]['file_name']
      mask = coco.annToMask(anns[0])
      for i in range(len(anns[1:])):
        if (anns[i]['category_id']==1):
          mask = np.bitwise_or(mask, coco.annToMask(anns[i]))
      yield [image_name, mask]

def create_coco_hdf5(path, num_images, image_shape):
  file = h5py.File(path, mode='w')
  images_dataset = file.create_dataset('images', (num_images, 3, image_shape[0], image_shape[1]))
  images_dataset.dims[0].label = 'batch'
  images_dataset.dims[1].label = 'channel'
  images_dataset.dims[2].label = 'height'
  images_dataset.dims[3].label = 'width'
  mask_data = file.create_dataset('masks', (num_images, image_shape[0], image_shape[1]))
  mask_data.dims[0].label = 'batch'
  mask_data.dims[1].label = 'height'
  mask_data.dims[2].label = 'width'
  image_embeddings = file.create_dataset('image_embeddings', (num_images, 256, 64, 64))
  image_embeddings.dims[0].label = 'batch'
  image_embeddings.dims[1].label = 'channel'
  image_embeddings.dims[2].label = 'width'
  image_embeddings.dims[3].label = 'height'
  return file, images_dataset, mask_data, image_embeddings

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

def coco_hdf5(data_path, hdf5_path, weights_path, image_shape):
  DEVICE = 'cuda'
  weights_path = weights_path +'mobile_sam.pt'

  encoder = MobileSAM(weights_path)
  encoder.to(device= DEVICE)
  encoder.eval()
  
  train_data = COCO(data_path + "/annotations/instances_train2017.json")
  train_len = sum([1 for _ in get_metadata(train_data)])

  # create hdf5 train datasets
  train_hdf5, train_images, train_masks, image_embeddings = create_coco_hdf5(hdf5_path + "train.hdf5", train_len, image_shape)

  # fill training dataset
  i = 0
  for image_name, mask in tqdm(get_metadata(train_data), total=train_len, desc="Image"):
    image = cv2.imread(data_path + "train_imgs/" + image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # cv2.imshow('image', image)
    # cv2.imshow('mask', (mask > 0.5).astype(np.uint8)*255.0)
    # cv2.waitKey(0)

    input = preprocess_image(image, image_shape)
    copy = input
    
    input = input.unsqueeze(dim=0).to(device=DEVICE, dtype=torch.float) / 255.0
    embedding = encoder(input)
    embedding = embedding.detach().cpu().numpy()
    image_embeddings[i] = embedding

    train_images[i] = copy

    mask = np.expand_dims(mask, axis=-1)
    mask = preprocess_image(mask, image_shape)
    train_masks[i] = mask
    i += 1

  # flush and close hdf5 files
  train_hdf5.flush()
  train_hdf5.close()

  print("Created training dataset! \n"
        "Size: %d \n"
        "Path: %s \n"
        % (train_len, data_path + "train.hdf5"))

def main():
  folder_path = os.getcwd()+"/"
  resources_path = folder_path + "resources/"
  data_path = resources_path + "data/"
  hdf5_path = resources_path + "hdf5/"
  weights_path = resources_path + "weights/"
  coco_hdf5(data_path, hdf5_path, weights_path, image_shape=(1024, 1024))

if __name__ == '__main__':
  main()
  