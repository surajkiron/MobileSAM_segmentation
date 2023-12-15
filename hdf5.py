import builtins
import sys
import h5py
import cv2
import glob
import os
import numpy as np
import torch
import imageio.v2 as imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from model import MobileSAM
from util import ResizeLongestSide
from config import parse_args, save_args


N_PRINT = 0
SEQUENCE_LENGTH = 4

# ----------------------------------------------------------------------------

def create_hdf5(path, num_images, maps_channels, maps_size, image_size):
    file = h5py.File(path, mode='w')
    input_data = file.create_dataset('maps', (num_images, maps_channels, maps_size, maps_size))
    input_data.dims[0].label = 'batch'
    input_data.dims[1].label = 'channels'
    input_data.dims[2].label = 'size'
    input_data.dims[3].label = 'size'
    label_data = file.create_dataset('label', (num_images, image_size, image_size))
    label_data.dims[0].label = 'batch'
    label_data.dims[1].label = 'size'
    label_data.dims[2].label = 'size'
    return file, input_data, label_data


# ----------------------------------------------------------------------------

def prepare_nyu(frames_folder, datapath):
    data = [] 
    for frame_file in os.listdir(frames_folder):
        with open(os.path.join(frames_folder, frame_file), 'r') as f: 
            triplets = f.readlines()

        scene_path = os.path.join(datapath, frame_file[:-4])
        for triplet in triplets:
            d_name, i_name, a_name = triplet.split(' ')
            imagepath = os.path.join(scene_path, i_name) 
            depthpath = os.path.join(scene_path, d_name)
            data.append((imagepath, depthpath))
    return data
    
# ----------------------------------------------------------------------------

def insert_str(string, str_to_insert, index):
    return string[:index] + str_to_insert + string[index:]

def prepare_kitti(imagepath_train, depthpath_train):
    data = []
    depthpaths = sorted(glob.glob(depthpath_train + "/**/*.png", recursive=True))
    imagepaths = [insert_str(p.replace("depths", "images").replace("proj_depth/groundtruth/", ""), 'data/', -14) for p in
    depthpaths]
    for i in range(len(imagepaths)):
        data.append([imagepaths[i], depthpaths[i]])
    return data


# ----------------------------------------------------------------------------

def colorize(value, vmin=None, vmax=None, cmap='viridis'):
    value = value[:, :, None]
    value = np.log10(value)
    vmax = value.max() if vmax is None else vmax
    value = np.clip(value, float(1e-3), vmax)
    vmin = value.min() if vmin is None else vmin
    
    if vmin != vmax: value = (value - vmin) / (vmax - vmin)
    else: value = value * 0.
    
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True).squeeze()
    img = value[:, :, :3]
    return img

# ----------------------------------------------------------------------------

def main():
    # parse arguments
    args = parse_args()

    image_size = 1024
    maps_channels = 256

    # paths
    weights_path = '/'.join(sys.path[0].split('/')[:-1]) + '/resources/weights/'

    # preprocessing
    transform = ResizeLongestSide(image_size)
    pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 3)
    pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, 3)

    # device
    args.device = 'cuda:0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print('Training model on cuda:' + str(args.gpu_id))
    # mobile sam
    encoder = MobileSAM(weights_path + 'mobile_sam.pt')
    encoder.to(device=args.device)
    encoder.eval()
   
    # define folder where to save hdf5 file
    hdf5_path = '/'.join(sys.path[0].split('/')[:-1]) + '/resources/hdf5/nyu_v2/'
    
    
    # load data
    data = os.listdir(nyu_path)
    data.sort()
    scene_list = list(set([x[:-11] for x in data]))
    # train_scene = scene_list[:-4]
    train_scene = scene_list
    train_data = [x for x in data if x[:-11] in train_scene]
    num_train = len(train_data)
    # valid_scene = scene_list[-4:-2]
    # valid_data = [x for x in data if x[:-11] in valid_scene]
    # num_valid = len(valid_data)
    # test_scene  = scene_list[-2:]
    # test_data = [x for x in data if x[:-11] in test_scene]
    # num_test = len(test_data)
    
    # instantiate hdf5 files
    maps_size = image_size // 16 
    train_hdf5, train_input, train_label, train_scene = create_hdf5(hdf5_path + 'train.hdf5', num_train, maps_channels,
            maps_size, image_size)
    print('Loaded', num_train, 'train data images ...')
    
    fig, axs = plt.subplots(2, 1)
    axs[0].axis('off')
    axs[1].axis('off')
    # fill training set
    idx = 0
    # for i in tqdm(range(num_train), total=num_train, desc='Train:'):
    for i in tqdm(range(10), total=(10), desc='Train:'):
        # image_path, depth_path = data[i] for kitti
        
        scene, scene_n, pair_n = train_data[i].split('_')
        pair_n = int(pair_n)
        if pair_n < SEQUENCE_LENGTH: continue

        for j in range(i-SEQUENCE_LENGTH): 
            img_dpt = h5py.File(os.path.join(nyu_path, train_data[i]))
        
        image = img_dpt['image'][:]
        image = np.transpose(image, (2, 1, 0))
        input = transform.apply_image(image)
        input = (input - pixel_mean) / pixel_std

        pad_h = (image_Wsize - input.shape[0]) // 2
        if input.shape[0] % 2: input = np.pad(input, ((pad_h, pad_h+1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        else: input = np.pad(input, ((pad_h, pad_h),   (0, 0), (0, 0)), mode='constant', constant_values=0)
        image = input
        
        input = np.transpose(input, (2, 0, 1))
        input = torch.as_tensor(input, dtype=torch.float, device=args.device)
        input = torch.unsqueeze(input, dim=0)
        input = encoder(input)
        input = input.detach().cpu().numpy()
        train_input[i] = input

        
        # depth = imageio.imread(depth_path).astype(np.float32) / 255. # kitti
        depth = img_dpt['depth'][:].astype(np.float32)
        depth = np.transpose(depth, (1, 0))
        label = transform.apply_image(depth)
        
        pad_h = (image_size - label.shape[0]) // 2
        if label.shape[0] % 2: label = np.pad(label, ((pad_h, pad_h+1), (0, 0)), mode='constant', constant_values=0)
        else: label = np.pad(label, ((pad_h, pad_h), (0, 0)), mode='constant', constant_values=0)
        train_label[i] = label
        
        # need of defining a scene name in order to pass meaningful sequences of frames to the network
        train_scene[i] = data[i][:-11]
        
        if i < N_PRINT:
            simage = (image/np.max(image)*255).astype(int)
            simage[simage<0] = 0
            axs[0].imshow(simage)
            slabel = (label/np.max(label)*255).astype(int)
            slabel[slabel<0] = 0
            axs[1].imshow(slabel)
            fig.savefig('../resources/trials/train_pairs_{}.png'.format(i))

    
    # flush and close hdf5 files
    train_hdf5.flush()
    train_hdf5.close()
    
    print('\nCreated training dataset!\nSize: %d\nPath: %s' % (num_train, hdf5_path + 'train.hdf5'))
    return 0

    # # instantiate hdf5 files
    # maps_size = image_size // 16
    # valid_hdf5, valid_input, valid_label = create_hdf5(hdf5_path + 'valid.hdf5', num_valid, maps_channels,
    #         maps_size, image_size)
    # print('Loaded', num_valid, 'validation data images ...')

    # # fill validation set
    # idx = 0
    # # for i in tqdm(range(num_valid), total=num_valid, desc='Validation:'):
    # for i in tqdm(range(10), total=10, desc='Validation:'):
    # #   image_path, depth_path = data_valid[i]
    #     img_dpt = h5py.File(os.path.join(nyu_path, data[i]))
    #     
    #     # image = image[:, :, ::-1]
    #     image = img_dpt['image'][:]
    #     image = np.transpose(image, (2, 1, 0))
    #     input = transform.apply_image(image)
    #     input = (input - pixel_mean) / pixel_std

    #     pad_h = (image_size - input.shape[0]) // 2
    #     if input.shape[0] % 2: input = np.pad(input, ((pad_h, pad_h+1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    #     else: input = np.pad(input, ((pad_h, pad_h),   (0, 0), (0, 0)), mode='constant', constant_values=0)
    #     image = input
    #     
    #     input = np.transpose(input, (2, 0, 1))
    #     input = torch.as_tensor(input, dtype=torch.float, device=args.device)
    #     input = torch.unsqueeze(input, dim=0)
    #     input = encoder(input)
    #     input = input.detach().cpu().numpy()
    #     valid_input[i] = input

    #     
    #     # depth = imageio.imread(depth_path).astype(np.float32) / 255. # kitti
    #     depth = img_dpt['depth'][:].astype(np.float32)
    #     depth = np.transpose(depth, (1, 0))
    #     label = transform.apply_image(depth)
    #     
    #     pad_h = (image_size - label.shape[0]) // 2
    #     if label.shape[0] % 2: label = np.pad(label, ((pad_h, pad_h+1), (0, 0)), mode='constant', constant_values=0)
    #     else: label = np.pad(label, ((pad_h, pad_h), (0, 0)), mode='constant', constant_values=0)
    #     valid_label[i] = label
    #     
    #     if i < N_PRINT:
    #         afig, axs = plt.subplots(2, 1) 
    #         simage = (image/np.max(image)*255).astype(int)
    #         simage[simage<0] = 0
    #         axs[0].imshow(simage)
    #         axs[0].axis('off')
    #         slabel = (label/np.max(label)*255).astype(int)
    #         slabel[slabel<0] = 0
    #         axs[1].imshow(slabel)
    #         axs[1].axis('off')
    #         fig.savefig('../resources/trials/valid_pairs_{}.png'.format(i))


    # # flush and close hdf5 files
    # valid_hdf5.flush()
    # valid_hdf5.close()
    # 
    # print('\nCreated validation dataset!\nSize: %d\nPath: %s' % (num_valid, hdf5_path + 'valid.hdf5'))

    # # instantiate hdf5 files
    # maps_size = image_size // 16
    # test_hdf5, test_input, test_label = create_hdf5(hdf5_path + 'test.hdf5', num_test, maps_channels,
    #         maps_size, image_size)
    # print('Loaded', num_test, 'testing data images ...')

    # # fill validation set
    # idx = 0
    # # for i in tqdm(range(num_test), total=num_test, desc='Testing:'):
    # for i in tqdm(range(10), total=10, desc='Testing:'):
    # #   image_path, depth_path = data_valid[i]
    #     img_dpt = h5py.File(os.path.join(nyu_path, data[i]))
    #     
    #     # image = image[:, :, ::-1]
    #     image = img_dpt['image'][:]
    #     image = np.transpose(image, (2, 1, 0))
    #     input = transform.apply_image(image)
    #     input = (input - pixel_mean) / pixel_std

    #     pad_h = (image_size - input.shape[0]) // 2
    #     if input.shape[0] % 2: input = np.pad(input, ((pad_h, pad_h+1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    #     else: input = np.pad(input, ((pad_h, pad_h),   (0, 0), (0, 0)), mode='constant', constant_values=0)
    #     image = input
    #     
    #     input = np.transpose(input, (2, 0, 1))
    #     input = torch.as_tensor(input, dtype=torch.float, device=args.device)
    #     input = torch.unsqueeze(input, dim=0)
    #     input = encoder(input)
    #     input = input.detach().cpu().numpy()
    #     test_input[i] = input

    #     
    #     # depth = imageio.imread(depth_path).astype(np.float32) / 255. # kitti
    #     depth = img_dpt['depth'][:].astype(np.float32)
    #     depth = np.transpose(depth, (1, 0))
    #     label = transform.apply_image(depth)
    #     
    #     pad_h = (image_size - label.shape[0]) // 2
    #     if label.shape[0] % 2: label = np.pad(label, ((pad_h, pad_h+1), (0, 0)), mode='constant', constant_values=0)
    #     else: label = np.pad(label, ((pad_h, pad_h), (0, 0)), mode='constant', constant_values=0)
    #     test_label[i] = label
    #     
    #     if i < N_PRINT:
    #         afig, axs = plt.subplots(2, 1) 
    #         simage = (image/np.max(image)*255).astype(int)
    #         simage[simage<0] = 0
    #         axs[0].imshow(simage)
    #         axs[0].axis('off')
    #         slabel = (label/np.max(label)*255).astype(int)
    #         slabel[slabel<0] = 0
    #         axs[1].imshow(slabel)
    #         axs[1].axis('off')
    #         fig.savefig('../resources/trials/test_pairs_{}.png'.format(i))


    # # # flush and close hdf5 files
    # test_hdf5.flush()
    # test_hdf5.close()
    # # 
    # print('\nCreated testing dataset!\nSize: %d\nPath: %s' % (num_test, hdf5_path + 'test.hdf5'))



# ----------------------------------------------------------------------------

if __name__ == '__main__':
    main()