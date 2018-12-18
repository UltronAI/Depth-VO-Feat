
import argparse
import collections
import scipy.io as sio
import numpy as np
import sys, glob, os, cv2, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as scid
import scipy.signal as scisig
import time
import unittest
from sklearn import preprocessing
from scipy.spatial.distance import pdist

caffe_root = '/home/gaof/caffe-dev/'
sys.path.insert(0, caffe_root + 'python')
import caffe

def gen_ResNetVlad():
    use_cuda = True
    
    model = ResNetVlad()
    if use_cuda:
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print('Multi_gpu')
            model = torch.nn.DataParallel(model)
    
    # state_dict = torch.load('vd16_data_wpca.pkl') # add map_location='cpu' if no gpu

    #state_dict = torch.load('./models/student_net_params_res18_init.pkl')
    # state_dict = torch.load('./outimport/student_net_params_epoch_479.pkl')
    state_dict = torch.load('../netvlad_distil/output_log/student_net_params_epoch_29.pkl')
    print(state_dict)

    model.load_state_dict(state_dict)

    model_single = model.module
    model_single = model_single.cpu()
    # print(model_single.state_dict() )
    torch.save(model_single.state_dict(), './output_log/student_net_paramssingle_29.pkl')
    # exit(0)

    for name,p in model_single.named_parameters():
        if 'WPCA.bias' in name:
            print(name,p)

    return model_single

root = '/home/gaof/workspace/Depth-VO-Feat/test_fix_point/'

def get_caffe_model():
    model_def = root + 'odometry_test.prototxt'
    caffe_model = root + 'Temporal.caffemodel'
    odom_net = caffe.Net(model_def, caffe_model, caffe.TEST)
    return odom_net

def get_fix_lable(filename1, filename2):
    with open(root + 'test_1.txt', 'w') as f:
        print("{} 0".format(filename1), file=f)
    with open(root + 'test_2.txt', 'w') as f:
        print("{} 0".format(filename2), file=f)
    assert(os.system('cd /home/gaof/workspace/Depth-VO-Feat/ && ./testsim.sh test') == 0)
    count = 0
    featureshape = np.array([16,80,304])
    features = np.zeros(featureshape.prod(),dtype=np.float32)
    with open("/home/gaof/workspace/Depth-VO-Feat/fix_results/fix_test.log") as f:                          # open txt file
        for line in f:                                          # check line by line
            datalist = line.split()                             # split one line to a list
            if len(datalist) > 8:                               # jump over void line 
                if datalist[3] == 'net_test.cpp:305]' and datalist[4] == 'Batch' and datalist[6] == 'conv_0_pose':
                    features[count] = np.float32( datalist[8] )
                    # print("input %d: %g" %(count,features[count]))
                    count += 1
    features = features.reshape(featureshape.tolist())
    return features    


def get_list_labls():
    # fileList='../image_2_flist.txt'
    # teaching = gen_ResNetVlad()
    file_root = '/home/gaof/workspace/00/image_2/'
    descs = []
    file_name_1 = file_root + '{:06}.png'.format(0)
    file_name_2 = file_root + '{:06}.png'.format(1)
    features = get_fix_lable(file_name_1, file_name_2)
    # TODO: run remaining caffe model with features as input.
    odom_net = get_caffe_model()
    odom_net.blobs['data'].data[0] = features
    odom_net.forward()
    output = odom_net.blobs['T_2to1'].data.copy()
    print(output)
    print(output.shape)
    descs = descs + output.tolist()
    """
    with open(fileList) as f:
        for line in f:
            
            line = line.strip()
            print(line)
            features = get_fix_lable(line)
            data = torch.from_numpy(features)
            data = data.unsqueeze(0)
            output = teaching(data)
            descs = descs + output.tolist()
    """
    return descs
            


def get_labels(batch_size):
    descs = []
    # teaching = gen_originNetVlad()
    teaching = gen_ResNetVlad()
    image_size = (382, 382)
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
        ])
    train_loader = torch.utils.data.DataLoader(
        ImageList(root='./image_2/', fileList='../image_2_flist.txt',transform=train_transform),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False)

    print('hello')
    for batch_idx, (data,imgPaths) in enumerate(train_loader):
        # print (batch_idx)
        # print ((data,imgPaths))
        iteration = batch_idx
        print(iteration)
        data = torch.round(data*255)
        data = data.cuda()
        # print(data.shape)
        output = teaching(data)
        
        

        descs = descs + output.tolist()

        # for index , jpeg_path in enumerate(imgPaths):
        #     print(index)
        #     print(jpeg_path)
        #     outputfile = 'teaching_result/' + jpeg_path.replace('.png','.pkl')
        #     torch.save(output[index],outputfile)
    return descs


def get_matrix(feats):
    use_dim = len(feats[0])
    print("length is {}".format(use_dim))
    use_feats = np.array(feats)[:, :use_dim]
    output = open(root + 'feats_00.pkl', 'wb')
    pickle.dump(feats, output)
    output.close()
    #score = np.dot(np.array(feats),np.array(feats).T)
    #sns.heatmap(score,annot=False,cmap='RdYlGn') #data.corr()-->correlation matrix
    #fig=plt.gcf()
    #fig.set_size_inches(12,12)
    #plt.savefig('euclidean_distance_'+LAYER_NAME)
    #plt.show()
    #output = open('./val_reloc/feats_score_00.pkl', 'wb')
    #pickle.dump(score, output)
    #output.close()
    #%% Suppressing a certain radius around the diagonal to prevent self-matches
    # (not real loop closures).
    # suppression_diameter = 501
    # diag_suppression = scisig.convolve2d(
    #         np.eye(use_feats.shape[0]), np.ones((1, suppression_diameter)), 'same')

    # #%% NetVLAD matching, reject along diagonal (query ~ match):
    # sq_dists = scid.squareform(scid.pdist(use_feats, 'sqeuclidean'))
    # sq_dists[diag_suppression > 0] = np.inf
    # plt.imshow(sq_dists)
    # plt.title('Confusion matrix NetVLAD')
    # plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    # get_labels('/home/yujc/mnt/google_landmark/index/google_landmarks/',32,'/home/yujc/mnt/netvlad_data/vlad_result/teaching_result/','/home/yujc/mnt/netvlad_data/vlad_result/teaching.list')
    descs = get_list_labls()
    # descs = get_labels(16)
    print(descs[0])
    get_matrix(descs)
    # train_list('image_2',3)
    # train_list('./image_2',8,'./teaching_result','teaching.list')
    # train_list('/home/yujc/mnt/google_landmark/index/google_landmarks/',32,'/home/yujc/mnt/netvlad_data/vlad_result/teaching_result/','/home/yujc/mnt/netvlad_data/vlad_result/teaching.list')
