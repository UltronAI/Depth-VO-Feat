# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch.utils.data as data
from path import Path
from scipy.misc import imresize, imread
from tqdm import tqdm
import random
import torch

class pose_framework_KITTI(data.Dataset):
    def __init__(self, root, sequence_set, step=1, transform=None, seed=2018, img_height=160, img_width=608, shuffle=True):
        np.random.seed(seed)
        random.seed(seed)
        self.shuffle = shuffle
        self.root, self.transform = root, transform
        self.img_files, self.poses = read_scene_data(self.root, sequence_set, step)
        self.sequence_num = len(self.poses)
        self.height, self.width = img_height, img_width
        self.generator()

    def generator(self):
        sequence_set = []
        self.gt_se3 = []
        for pose_list in self.poses:
            prev_SE3 = np.eye(4)
            se3_list = []
            for idx in range(pose_list.shape[0]):
                SE3 = np.eye(4)
                SE3[:3] = pose_list[idx]
                se3_list.append(np.linalg.inv(prev_SE3).dot(SE3))
                prev_SE3 = SE3
            self.gt_se3.append(se3_list)
        for img_list, pose_list in zip(self.img_files, self.gt_se3):
            #print(pose_list.shape)
            #exit(0)
            for i in range(len(img_list)-1):
                imgs = [img_list[i], img_list[i+1]]
                pose = pose_list[i+1]

                sample = {'imgs': imgs, 'pose': pose}
                sequence_set.append(sample)
        if self.shuffle:
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [imread(img).astype(np.float32) for img in sample['imgs']]
        imgs = [imresize(img, (self.height, self.width)).astype(np.float32) for img in imgs]
        if self.transform is not None:
            imgs = self.transform(imgs)
            img_data = np.zeros((6, self.height, self.width)).astype(np.float32)
            imgs_1 = imgs[0].numpy()
            imgs_1[0] -= 101
            imgs_1[1] -= 117
            imgs_1[2] -= 123
            imgs_2 = imgs[1].numpy()
            imgs_2[0] -= 101
            imgs_2[1] -= 117
            imgs_2[2] -= 123
            img_data[:3] = imgs_2
            img_data[3:] = imgs_1
        else:
            img_data = np.zeros((6, self.height, self.width)).astype(np.float32)
            imgs_1 = np.transpose(imgs[0], (2,0,1))
            imgs_1[0] -= 101
            imgs_1[1] -= 117
            imgs_1[2] -= 123
            imgs_2 = np.transpose(imgs[1], (2,0,1))
            imgs_2[0] -= 101
            imgs_2[1] -= 117
            imgs_2[2] -= 123
            img_data[:3] = imgs_2
            img_data[3:] = imgs_1
        return torch.from_numpy(img_data).type(torch.FloatTensor), sample['pose']

    def __len__(self):
        return len(self.samples)


def read_scene_data(data_root, sequence_set, step=1):
    data_root = Path(data_root)
    im_sequences = []
    poses_sequences = []

    sequences = set()
    for seq in tqdm(sequence_set, leave=False):
        corresponding_dirs = set((data_root/'sequences').dirs(seq))
        sequences = sequences | corresponding_dirs

    print('getting test metadata for these sequences : {}'.format(sequences))
    for sequence in sequences:
        poses = np.genfromtxt(data_root/'poses'/'{}.txt'.format(sequence.name)).astype(np.float32).reshape(-1, 3, 4)
        imgs = sorted((sequence/'image_2').files('*.png'))
        im_sequences.append(imgs)
        poses_sequences.append(poses)
    return im_sequences, poses_sequences
