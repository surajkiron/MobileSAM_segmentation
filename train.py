import sys
import os
import time
import warnings
import pytorch_lightning

from config import parse_args, save_args
from util import check_folder_paths
from data import load_dataset
from lightning import HumanDetectionNetwork

warnings.filterwarnings('ignore')

#----------------------------------------------------------------------------

if __name__ == '__main__':
  # parse arguments
  args = parse_args()

  # set global paths
  folder_path = os.getcwd() + "/"
  resources_path = folder_path + 'resources/'
  hdf5_path = resources_path + 'hdf5/'
  experiment_path = resources_path + 'experiments/' + time.strftime('%Y%m%d-%H%M%S') + '_' + str(args.run_id) + '/'
  check_folder_paths([experiment_path + 'checkpoints', experiment_path + 'predictions'])
 
  # device
  args.device = 'cuda:0'
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
  print('Training model on cuda:' + str(args.gpu_id))
  
  # save arguments
  save_args(args, experiment_path + 'args.txt')

  # checkpoint
  checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(monitor="train_loss",
                                                                    dirpath=experiment_path + 'checkpoints',
                                                                    filename="best_model",
                                                                    save_last=True,
                                                                    mode="min",
                                                                    verbose=True)
  checkpoint_callback.CHECKPOINT_NAME_LAST = "last_model"
  checkpoint_callback.FILE_EXTENSION = ".pth"

  # load training data
  train_dataset, train_loader = load_dataset(mode='training',
                                             file_path=hdf5_path + 'train.hdf5',
                                             batch_size=args.batch_size,
                                             shuffle=args.shuffle,
                                             num_workers=16,
                                             pin_memory=True)
  
  valid_dataset, valid_loader = load_dataset(mode='validating',
                                            file_path=hdf5_path + 'val.hdf5',
                                            batch_size=args.batch_size,
                                            shuffle=args.shuffle,
                                            num_workers=16,
                                            pin_memory=True)

  network = HumanDetectionNetwork(args.lr)
  trainer = pytorch_lightning.Trainer(accelerator='gpu',
                                      devices=args.num_devices,
                                      precision=args.precision,
                                      max_epochs=args.epochs,
                                      check_val_every_n_epoch=args.valid_frequency,
                                      logger=True,
                                      callbacks=[checkpoint_callback])
  trainer.fit(network, train_loader, val_dataloaders=valid_loader)
