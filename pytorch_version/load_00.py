# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch.utils.data as data
from path import Path
from scipy.misc import imresize, imread
from tqdm import tqdm
import torch

class pose_framework_KITTI(data.Dataset):
    def __init__(self, root, filetxt, transform=None, img_height=160, img_width=608):
        self.root, self.transform = root, transform
        self.img_files, self.poses = self.read_scene_data(self.root, filetxt)
        self.height, self.width = img_height, img_width
        self.generator()

    def generator(self):
        prev_SE3 = np.eye(4)
        se3_list = []
        for idx in range(self.poses.shape[0]):
            SE3 = np.eye(4)
            SE3[:3] = self.poses[idx]
            se3_list.append(np.linalg.inv(prev_SE3).dot(SE3))
            prev_SE3 = SE3
        self.gt_se3 = se3_list

    def __getitem__(self, index):
        imgs_path = [self.img_files[index], self.img_files[index+1]]
        pose = self.gt_se3[index + 1]
#        print("{}, {}".format(imgs_path[0], imgs_path[1]))
        imgs = [imread(self.data_root/'sequences'/'09'/'image_2'/img).astype(np.float32) for img in imgs_path]
        imgs = [imresize(img, (self.height, self.width)).astype(np.float32) for img in imgs]
        if self.transform is not None:
#            print(imgs[0].shape)
            imgs = self.transform(imgs)
#            print(imgs[0].numpy().shape)
#            exit(0)
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
            imgs_1 = imgs[0]
            imgs_1[0] -= 101
            imgs_1[1] -= 117
            imgs_1[2] -= 123
            imgs_2 = imgs[1]
            imgs_2[0] -= 101
            imgs_2[1] -= 117
            imgs_2[2] -= 123
            img_data[:3] = imgs_2
            img_data[3:] = imgs_1
        return torch.from_numpy(img_data).type(torch.FloatTensor), torch.from_numpy(pose).type(torch.FloatTensor)

    def __len__(self):
        return len(self.gt_se3) - 1


    def read_scene_data(self, data_root, filetxt):
        self.data_root = Path(data_root)

        poses = np.genfromtxt(self.data_root/'poses'/'09.txt').astype(np.float32).reshape(-1, 3, 4)
        imgs = [line.strip() for line in open(filetxt, 'r').readlines()]
        return imgs, poses
