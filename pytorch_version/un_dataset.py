# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch.utils.data as data
from path import Path
from scipy.misc import imresize, imread
from tqdm import tqdm
import random
import torch

class dataset(data.Dataset):
    def __init__(self, transform=None, seed=9999, img_height=160, img_width=608, shuffle=True):
        np.random.seed(seed)
        random.seed(seed)
        self.shuffle = shuffle
        self.transform = transform
        self.height, self.width = img_height, img_width
        self.generator()

    def generator(self):
        f = open('/home/gaof/workspace/Depth-VO-Feat/data/kitti_eigen/train.txt', 'r')
        root = '/home/gaof/workspace/Depth-VO-Feat/data/kitti_eigen/'
        lines = f.readlines()
        self.samples = []
        for i in range(len(lines)):
            line_split = lines[i].split()
            l1, l2, r1, r2, k, t = line_split[0], line_split[1], line_split[2], line_split[3], line_split[4], line_split[5]
            intrinsics = root + 'intrinsics/' + l1.split('/')[-4] + '_cam.txt'
            raw_K = root + 'train_K/' + k + '.npy'
            T = root + 'train_T_R2L/' + t + '.npy'
            self.samples.append({
                'left_1': l1, 
                'left_2': l2,
                'right_1': r1,
                'right_2': r2,
                'intrinsics': intrinsics,
                'raw_K': raw_K,
                'T_R2L': T
            })

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [imread(img).astype(np.float32) for img in [sample['right_1'], sample['left_2'], sample['right_2']]]
        imgs = [imresize(img, (self.height, self.width)).astype(np.float32) for img in imgs]
        img_r1, img_l2, img_r2 = imgs[0], imgs[1], imgs[2]
        intrinsics = np.genfromtxt(sample['intrinsics']).astype(np.float32).reshape((3, 3))
        raw_K = np.load(sample['raw_K']).astype(np.float32)
        T_R2L = np.load(sample['T_R2L']).astype(np.float32)
        if self.transform is not None:
            #intrinsics = self.transform(intrinsics).numpy()
            img_r1 = self.transform(img_r1).numpy()
            img_r1 -= 101
            img_r1 -= 117
            img_r1 -= 123
            img_l2 = self.transform(img_l2).numpy()
            img_l2 -= 101
            img_l2 -= 117
            img_l2 -= 123
            img_r2 = self.transform(img_r2).numpy()
            img_r2 -= 101
            img_r2 -= 117
            img_r2 -= 123
        else:
            img_r1 = np.transpose(img_r1, (2,0,1))
            img_r1 -= 101
            img_r1 -= 117
            img_r1 -= 123
            img_l2 = np.transpose(img_l2, (2,0,1))
            img_l2 -= 101
            img_l2 -= 117
            img_l2 -= 123
            img_r2 = np.transpose(img_r2, (2,0,1))
            img_r2 -= 101
            img_r2 -= 117
            img_r2 -= 123
        return torch.from_numpy(img_r1).type(torch.FloatTensor), \
            torch.from_numpy(img_l2).type(torch.FloatTensor), \
            torch.from_numpy(img_r2).type(torch.FloatTensor), \
            torch.from_numpy(intrinsics).type(torch.FloatTensor), \
            torch.from_numpy(np.linalg.inv(intrinsics)).type(torch.FloatTensor), \
            torch.from_numpy(raw_K).type(torch.FloatTensor)
            torch.from_numpy(T_R2L).type(torch.FloatTensor)

    def __len__(self):
        return len(self.samples)
