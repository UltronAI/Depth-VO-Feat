#!/usr/bin/env python
import numpy as np
import sys
import numpy as np
from matplotlib import pyplot as plt

caffe_root = '/home/gaof/caffe-dev/'
sys.path.insert(0, caffe_root + 'python')
import caffe

import h5py
import os, os.path
import cv2
import argparse

img_width = 608;
img_height = 160;

root = '/home/gaof/workspace/Depth-VO-Feat/test_fix_point/'

def get_caffe_model():
    model_def = root + 'odometry_test.prototxt'
    caffe_model = root + 'Full-NYUv2.caffemodel'
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

def main():
    model_def_1 = "/home/gaof/workspace/Depth-VO-Feat/experiments/networks/odometry_deploy.prototxt"
#    model_def_2 = "/home/gaof/workspace/Depth-VO-Feat/test_fix_point/odometry_deploy_img.prototxt"
    model_def_2 = "/home/gaof/workspace/Depth-VO-Feat/experiments/networks/odometry_deploy_img.prototxt"

    caffe_model = "/home/gaof/workspace/Depth-VO-Feat/test_fix_point/Full-NYUv2.caffemodel"

    odom_net_1 = caffe.Net(model_def_1, caffe_model, caffe.TEST)
    odom_net_2 = caffe.Net(model_def_2, caffe_model, caffe.TEST)
    result_path = "/home/gaof/workspace/Depth-VO-Feat/odometry_results"

#    img1_path = "/home/gaof/workspace/00/image_2/000000.png"
#    img2_path = "/home/gaof/workspace/00/image_2/000001.png"

#    img1 = getImage(img1_path, True)
#    img2 = getImage(img2_path, True)

    odom_net_2.forward()
#    print(odom_net_2.blobs['conv_5_pose'].data.copy().shape)
#    print(odom_net_2.blobs['fc_0_pose'].data.copy().shape) 
    print(odom_net_2.blobs['T_2to1'].data.copy())
    #pred_poses_2 = odom_net_2.blobs['SE3'].data.copy()

    #pred_poses_1 = np.array(get_list_labels())
    #print(pred_poses_1.shape)
    #print(pred_poses_2.shape)
    #print('abs error: {}'.format(np.sum(np.abs(pred_poses_1-pred_poses_2))))

if __name__ == "__main__":
    main()
