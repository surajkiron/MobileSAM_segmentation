import os
import warnings
import torch
import pytorch_lightning

from config import parse_args, save_args
from util import check_folder_paths
from data import load_dataset
from lightning import HumanDetectionNetwork

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------

if __name__ == '__main__':
  # parse arguments
  args = parse_args()

  # Seed
  pytorch_lightning.seed_everything(args.seed)

  # set global paths
  folder_path = os.getcwd() + '/'
  resources_path = folder_path + 'resources/'
  hdf5_path = resources_path + 'hdf5/'
  experiment_path = resources_path + 'experiments/' + args.weights_path + '/'
  assert os.path.exists(experiment_path), 'Please specify weights path for loading model'
  check_folder_paths([experiment_path + 'testing'])

  # device
  args.device = 'cuda:0'
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
  print('Training model on cuda:' + str(args.gpu_id))

  # logging
  wandb_logger = pytorch_lightning.loggers.WandbLogger(project="image_dynamics", entity="vg-ssl")

  # load data
  test_dataset, test_loader = load_dataset(mode='testing',
                                            file_path=hdf5_path + 'val.hdf5',
                                            batch_size=args.batch_size,
                                            shuffle=args.shuffle,
                                            num_workers=16,
                                            pin_memory=True)

  # create model
  sample_data = next(iter(test_loader))
  network = HumanDetectionNetwork(args.lr)

  # load saved weights
  weight = torch.load(experiment_path + 'checkpoints/best_model.pth')['state_dict']
  # new_weight = dict()
  # for k, v in weight.items():
    # new_weight[k.replace('mlp.', '')] = weight[k]
  # network.mlp.load_state_dict(new_weight, strict=False)

  # test model
  trainer = pytorch_lightning.Trainer(accelerator='gpu',
                                      devices=args.num_devices,
                                      precision=args.precision,
                                      max_epochs=args.epochs,
                                      check_val_every_n_epoch=args.valid_frequency,
                                      logger=True)
  # if trainer.is_global_zero:
    # wandb_logger.experiment.config.update(vars(args))
  results = trainer.test(network, test_loader)