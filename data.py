import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset

#----------------------------------------------------------------------------

class MobileSAMDataset(Dataset):
  def __init__(self, hdf5_path):
    self.hdf5_path = hdf5_path
    self.data = None

    # load info of hdf5
    data_file = h5py.File(hdf5_path, 'r')
    self.data_len = len(data_file['images'])
    self.batch = data_file['images'][()][0].shape[0]
    self.height = data_file['images'][()][0].shape[1]
    self.width = data_file['images'][()][0].shape[2]

    # self.mask_len = len(data_file['masks'])
    # self.mask_height     = data_file['masks'][()][0].shape[0]
    # self.mask_width    = data_file['masks'][()][0].shape[1]

    # self.embed_len = len(data_file['image_embeddings'])
    # self.embed_channel = data_file['image_embeddings'][()][0].shape[0]
    # self.embed_width     = data_file['image_embeddings'][()][0].shape[1]
    # self.embed_height  = data_file['image_embeddings'][()][0].shape[2]

  def __len__(self):
    return self.data_len

  def __getitem__(self, i):
    if self.data is None:
      self.data = h5py.File(self.hdf5_path, 'r')
    input = self.data['image_embeddings'][i]
    label = self.data['masks'][i]
    return input, label

#----------------------------------------------------------------------------

def load_dataset(mode, file_path, batch_size, shuffle, num_workers, pin_memory):
  print('Generating', mode, 'data ...')
  dataset = MobileSAMDataset(file_path)
  batch_size = batch_size if batch_size > 0 else len(dataset)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
  print('... Loaded', dataset.data_len, 'images')
  print('|Image| = (%i, %i, %i)' % (dataset.batch, dataset.height, dataset.width))
  print('|Mask| = (%i, %i)' % (dataset.height, dataset.width))
  return dataset, loader
