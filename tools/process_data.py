import numpy as np
import sys
import argparse
import os, os.path
import cv2

parser = argparse.ArgumentParser(description="Process data for quantization")
parser.add_argument('--output', type=str, default='.')
parser.add_argument('--type', type=str, default='input')

global args
args = parser.parse_args()
if not os.path.exists(args.output):
    os.makedirs(args.output)

dataset_root = '/home/share/kitti/odometry/'

def getImage(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print "img_path: ", img_path
        assert img!=None, "Image reading error!"
    img = cv2.resize(img, (608, 160))
    img = img.transpose((2,0,1))
    img = img.astype(np.float32)
    img[0] -= 104
    img[1] -= 117
    img[2] -= 123
    return img

def preprocessData():
    print "Prepocess input data ..."
    seq_path = dataset_root + 'sequences/00/image_2/'
    start_idx, end_idx = 0, 100
    for idx in xrange(start_idx, end_idx):
        img1_path = seq_path + '{:06}.png'.format(idx)
        img2_path = seq_path + '{:06}.png'.format(idx+1)
        img1 = getImage(img1_path)
        img2 = getImage(img2_path)
        data = np.zeros((1, 6, 160, 608))
        data[0, :3] = img2
        data[0, 3:] = img1
        np.save(os.path.join(args.output, '{:06}.npy'.format(idx)), data)

def processTarget():
    print "Process target ..."
    gt_path = dataset_root + 'poses/00.txt'
    gt_lines = open(gt_path, 'r').readlines()
    start_idx, end_idx = 0, 100
    gt_SE3 = []
    prev_SE3 = np.eye(4)
    for idx in xrange(start_idx, end_idx):
        line = gt_lines[idx].strip('\n').split()
        SE3 = np.zeros((4, 4))
        SE3[0,0] = float(line[0])
        SE3[0,1] = float(line[1])
        SE3[0,2] = float(line[2])
        SE3[0,3] = float(line[3]) 
        SE3[1,0] = float(line[4])
        SE3[1,1] = float(line[5])
        SE3[1,2] = float(line[6])
        SE3[1,3] = float(line[7])
        SE3[2,0] = float(line[8])
        SE3[2,1] = float(line[9])
        SE3[2,2] = float(line[10])
        SE3[2,3] = float(line[11])
        SE3[3,3] = 1
        np.save(os.path.join(args.output, '{:06}.npy'.format(idx)), np.linalg.inv(prev_SE3).dot(SE3))
        prev_SE3 = SE3
        
if __name__ == '__main__':
    if args.type == 'input':
        preprocessData()
    else:
        processTarget()
