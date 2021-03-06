# -*- coding: utf-8 -*-

from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from fixmodel import FixOdometryNet
from PoseExpNet import PoseExpNet
# from model import DepthNet
from DispNetS import DispNetS
from feat_extractor import FeatExtractor
import cv2, os
import numpy as np
from path import Path

from tqdm import tqdm
import nics_fix_pt as nfp
import argparse
import pose_transforms
from dataset import pose_framework_KITTI
from un_dataset import dataset
from se3_generate import *
from loss_function_sfm import photometric_reconstruction_loss, smooth_loss, explainability_loss

# hyper-parameters
BITWIDTH = 8

# argparser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--odometry', default=None, type=str)
parser.add_argument('--depth', default=None, type=str)
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-8, metavar='WD',
                    help='SGD weight decay (default: 1e-8)')
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 2019)')
parser.add_argument('-b', '--batch-size', default=64, type=int)

parser.add_argument('-g', '--gpu-id', type=int, metavar='N', default=-1)
parser.add_argument("--dataset-dir", default='/home/share/kitti_odometry/dataset/', type=str, help="Dataset directory")
parser.add_argument("--train-sequences", default=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], 
                    type=str, nargs='*', help="sequences to train")
parser.add_argument("--test-sequences", default=['00'], type=str, nargs='*', help="sequences to test")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--log-interval', type=int, default=10, metavar='N')
parser.add_argument('--output-dir', type=str, default='./checkpoints')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")        
data_parallel = False

best_val_loss = 99999

def train(odometry_net, depth_net, feat_extractor, train_loader, epoch, optimizer):
    global device
    global data_parallel
    odometry_net.train()
    depth_net.train()
    feat_extractor.train()
    total_loss = 0
    img_reconstruction_total = 0
    exp_total = 0
    smooth_total = 0
    for batch_idx, (img_R1, img_L2, img_R2, intrinsics, inv_intrinsics, raw_K, T_R2L) in tqdm(enumerate(train_loader), desc='Train epoch %d' % epoch, leave=False, ncols=80):
        img_R1 = img_R1.type(torch.FloatTensor).to(device)
        img_R2 = img_R2.type(torch.FloatTensor).to(device)
        img_L2 = img_L2.type(torch.FloatTensor).to(device)
        intrinsics = intrinsics.type(torch.FloatTensor).to(device)
        inv_intrinsics = inv_intrinsics.type(torch.FloatTensor).to(device)
        raw_K = raw_K.type(torch.FloatTensor).to(device)
        T_R2L = T_R2L.type(torch.FloatTensor).to(device)

        img_R = torch.cat((img_R2, img_R1), dim=1)

        inv_depth_img_R2 = depth_net(img_R2)
        depth = [1 / (disp + 1e-4) for disp in inv_depth_img_R2]

        mask, T_2to1 = odometry_net(img_R)
        T_2to1 = T_2to1.view(T_2to1.size(0), -1)
        T_R2L = T_R2L.view(T_R2L.size(0), -1)

        img_reconstruction_error = photometric_reconstruction_loss(img_R2, img_R1, img_L2, depth, T_2to1, T_R2L, mask, intrinsics, inv_intrinsics)
        smooth_error = smooth_loss(depth, scale_factor=2.3)
        exp_loss = explainability_loss(mask)

        loss = img_reconstruction_error +  0.01 * smooth_error + 0.05 * exp_loss

        total_loss += loss.item()
        img_reconstruction_total += img_reconstruction_error.item()
        exp_total += exp_loss.item()
        smooth_total += smooth_error.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Train epoch {}: loss: {:.9f} photo-loss: {:.9f} exp-loss: {:.9f} smooth-loss: {:.9f}".format(epoch, 
        total_loss/len(train_loader), img_reconstruction_total/len(train_loader), exp_total/len(train_loader), smooth_total/len(train_loader)))


@torch.no_grad()
def validate(model, depth_net, feat_extractor, val_loader, epoch, output_dir, use_float=False):
    global device
    global best_val_loss
    global data_parallel
    model.eval()
    val_loss = 0
    for data, target in val_loader:
        data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
        mask, output = model(data)
        output = generate_se3(output.view(-1, 6, 1, 1))
        output = output.view(-1, 4, 4).type(torch.FloatTensor).to(device)
        val_loss += F.l1_loss(output, target).item()# sum up batch loss

    val_loss /= len(val_loader.dataset)
    is_best = val_loss < best_val_loss
    if not use_float:
        best_val_loss = val_loss if is_best else best_val_loss
    print('Test set: Average loss: {:.6f} [BEST:{}]'.format(val_loss, is_best if not use_float else 'N/A'))
    if use_float:
        return
    if is_best and args.gpu_id in range(4):
        torch.save(model.state_dict(), output_dir/"best_vo_checkpoint.pth.tar")
        torch.save(depth_net.state_dict(), output_dir/"best_depth_checkpoint.pth.tar")
        torch.save(feat_extractor.state_dict(), output_dir/"best_feat_checkpoint.pth.tar")
    elif is_best:
        torch.save(model.module.state_dict(), output_dir/"best_vo_checkpoint.pth.tar")
        torch.save(depth_net.module.state_dict(), output_dir/"best_depth_checkpoint.pth.tar")
        torch.save(feat_extractor.module.state_dict(), output_dir/"best_feat_checkpoint.pth.tar")

def main():
    global device
    global data_parallel
    print("=> will save everthing to {}".format(args.output_dir))
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    # Data loading code
    train_transform = pose_transforms.Compose([
        pose_transforms.RandomHorizontalFlip(),
        pose_transforms.ArrayToTensor()
    ])
    valid_transform = pose_transforms.Compose([pose_transforms.ArrayToTensor()])

    print("=> fetching sequences in '{}'".format(args.dataset_dir))
    dataset_dir = Path(args.dataset_dir)
    print("=> preparing train set") 
    train_set = dataset()#transform=train_transform)
    print("=> preparing val set")
    val_set = pose_framework_KITTI(
        dataset_dir, args.test_sequences, 
        transform=valid_transform,
        seed=args.seed, shuffle=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    odometry_net = PoseExpNet().to(device)
    depth_net = DispNetS().to(device)
    feat_extractor = FeatExtractor().to(device)

    # init weights of model
    if args.odometry is None:
        odometry_net.init_weights()
    elif args.odometry:
        weights = torch.load(args.odometry)
        odometry_net.load_state_dict(weights)
    if args.depth is None:
        depth_net.init_weights()
    elif args.depth:
        weights = torch.load(args.depth)
        depth_net.load_state_dict(weights['state_dict'])

    feat_extractor.init_weights()

    cudnn.benchmark = True
    if args.cuda and args.gpu_id in range(2):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    elif args.cuda:
        data_parallel = True
        odometry_net = torch.nn.DataParallel(odometry_net)
        depth_net = torch.nn.DataParallel(depth_net)
        feat_extractor = torch.nn.DataParallel(feat_extractor)

    optim_params = [
        {'params': odometry_net.parameters(), 'lr': args.lr},
        {'params': depth_net.parameters(), 'lr': args.lr},
        {'params': feat_extractor.parameters(), 'lr': args.lr}
    ]

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer = optim.Adam(optim_params, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    print("=> validating before training")
    #validate(odometry_net, depth_net, val_loader, 0, output_dir, True)
    print("=> training & validating")
    #validate(odometry_net, depth_net, val_loader, 0, output_dir)
    for epoch in range(1, args.epochs+1):
        train(odometry_net, depth_net, feat_extractor, train_loader, epoch, optimizer)
        validate(odometry_net, depth_net, feat_extractor, val_loader, epoch, output_dir)

if __name__ == '__main__':
    main()
    
    

