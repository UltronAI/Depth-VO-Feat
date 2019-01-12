#!/usr/bin/env python
import numpy as np
import sys
import numpy as np
from matplotlib import pyplot as plt

caffe_root = '/home/gaof/caffe-comp/'
sys.path.insert(0, caffe_root + 'python')
import caffe

import h5py
import os
import cv2
import argparse

img_width = 608;
img_height = 160;

root = '/home/gaof/workspace/Depth-VO-Feat/test_fix_point/'

def get_caffe_model(model_def, caffe_model):
    model_def = os.path.join(root, model_def)
    caffe_model = os.path.join(root, caffe_model)
    odom_net = caffe.Net(model_def, caffe_model, caffe.TEST)
    return odom_net

def get_fix_label(filename1, filename2, shape):
    with open(root + 'test_1.txt', 'w') as f:
        print("{} 0".format(filename1), file=f)
    with open(root + 'test_2.txt', 'w') as f:
        print("{} 0".format(filename2), file=f)
    assert(os.system('cd /home/gaof/workspace/Depth-VO-Feat/ && ./testsim.sh test') == 0)
    count = 0
    featureshape = np.array(shape)
    features = np.zeros(featureshape.prod(),dtype=np.float32)
    with open("/home/gaof/workspace/Depth-VO-Feat/fix_results/fix_test.log") as f:                          # open txt file
        for line in f:                                          # check line by line
            datalist = line.split()                             # split one line to a list
            if len(datalist) > 8:                               # jump over void line 
                if datalist[3] == 'net_test.cpp:305]' and datalist[4] == 'Batch' and datalist[6] == 'conv_2_pose':
                    try:
                        features[count] = np.float32( datalist[8] )
                    except:
                        print(count)
                    # print("input %d: %g" %(count,features[count]))
                    count += 1
    features = features.reshape(featureshape.tolist())
    return features    


def get_list_labels():
    # fileList='../image_2_flist.txt'
    # teaching = gen_ResNetVlad()
    file_root = '/home/gaof/workspace/00/image_2/'
    descs = []
    file_name_1 = file_root + '{:06}.png'.format(0)
    file_name_2 = file_root + '{:06}.png'.format(1)
    features = get_fix_label(file_name_1, file_name_2, [64, 20, 76])
    odom_net = get_caffe_model()
    odom_net.blobs['data'].data[0] = features
    odom_net.forward()
    output = odom_net.blobs['T_2to1'].data.copy()
    #print(output)
    #print(output.shape)
    descs = descs + output.tolist()
    return descs

def get_matrix(feats):
    use_dim = len(feats[0])
    print("length is {}".format(use_dim))
    use_feats = np.array(feats)[:, :use_dim]
    output = open(root + 'feats_00.pkl', 'wb')
    pickle.dump(feats, output)
    output.close()
    #score = np.dot(np.array(feats),np.array(feats).T)

def getImage(img_path, transform=True): 
    # ----------------------------------------------------------------------
    # Get and preprocess image
    # ----------------------------------------------------------------------
    img = cv2.imread(img_path)
    if img is None:
        print("img_path: " + img_path)
        assert img is not None, "Image reading error. Check whether your image path is correct or not."
    img = cv2.resize(img, (img_width, img_height))
    img = img.transpose((2,0,1))
    img = img.astype(np.float32)
    if transform:
        img[0] -= 104
        img[1] -= 117
        img[2] -= 123
    return img

def SE3_cam2world(pred_poses):
    pred_SE3_world = []
    cur_t = np.eye(4)
    pred_SE3_world.append(cur_t)
    for pose in pred_poses:
        cur_t = np.dot(cur_t, pose)
        pred_SE3_world.append(cur_t)
    return pred_SE3_world

def save_results(pred_poses, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'caffe_pred_1.txt'), 'w') as f:
        for se3 in pred_poses:
            tx = str(se3[0,3])
            ty = str(se3[1,3])
            tz = str(se3[2,3])
            R00 = str(se3[0,0])
            R01 = str(se3[0,1])
            R02 = str(se3[0,2])
            R10 = str(se3[1,0])
            R11 = str(se3[1,1])
            R12 = str(se3[1,2])
            R20 = str(se3[2,0])
            R21 = str(se3[2,1])
            R22 = str(se3[2,2])
            line_to_write = " ".join([R00, R01, R02, tx, R10, R11, R12, ty, R20, R21, R22, tz])
            f.writelines(line_to_write + '\n')

def main():
    caffe_model = "/home/gaof/workspace/Depth-VO-Feat/test_fix_point/Full-NYUv2.caffemodel"

    odom_net_float = get_caffe_model('odometry_deploy_data.prototxt', 'Full-NYUv2.caffemodel')
    odom_net_fix = get_caffe_model('odometry_deploy_img.prototxt', 'Full-NYUv2.caffemodel')
    result_path = "/home/gaof/workspace/Depth-VO-Feat/results"
    data_root = "/home/share/kitti_odometry/dataset/sequences/00/image_2"

    pred_poses = []
    with open('kitti_00.txt') as imgfile:
        lines = imgfile.readlines()
        for i in range(len(lines)-1):
            img1 = getImage(os.path.join(data_root, lines[i].strip()))
            img2 = getImage(os.path.join(data_root, lines[i+1].strip()))
            odom_net_1.blobs['data'].data[0, :3] = img2
            odom_net_1.blobs['data'].data[0, 3:] = img1
            odom_net_1.forward()
            odom_net_2.forward()
            assert (odom_net_2.blobs['SE3'].data[0,0] == odom_net_1.blobs['SE3'].data[0,0]).all()
            pred_poses.append(odom_net_1.blobs['SE3'].data[0,0].copy())
#    se3 = SE3_cam2world(pred_poses) 
    save_results(pred_poses, result_path)
      
    #pred_poses_2 = odom_net_2.blobs['SE3'].data.copy()
    #pred_poses_1 = np.array(get_list_labels())
    #print(pred_poses_1.shape)
    #print(pred_poses_2.shape)
    #print('abs error: {}'.format(np.sum(np.abs(pred_poses_1-pred_poses_2))))

if __name__ == "__main__":
    main()
