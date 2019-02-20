# -*- coding: utf-8 -*-

from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from fixmodel import FixOdometryNet
from model import DepthNet
from DispNetS import DispNetS
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
from geo_transform import geo_transform, pin_hole_project, inverse_warp
from loss_functions import smooth_loss

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

best_val_loss = 99999

def train(odometry_net, depth_net, train_loader, epoch, optimizer):
    global device
    odometry_net.set_fix_method(nfp.FIX_AUTO)
    odometry_net.train()
    depth_net.train()
    total_loss = 0
    lr_total = 0
    r12_total = 0
    smooth_total = 0
    for batch_idx, (img_R1, img_L2, img_R2, intrinsics, inv_intrinsics, raw_K, T_R2L) in tqdm(enumerate(train_loader), desc='Train epoch %d' % epoch, leave=False, ncols=80):
        img_R1 = img_R1.type(torch.FloatTensor).to(device)
        img_R2 = img_R2.type(torch.FloatTensor).to(device)
        img_L2 = img_L2.type(torch.FloatTensor).to(device)
        intrinsics = intrinsics.type(torch.FloatTensor).to(device)
        inv_intrinsics = inv_intrinsics.type(torch.FloatTensor).to(device)
        raw_K = raw_K.type(torch.FloatTensor).to(device)
        T_R2L = T_R2L.type(torch.FloatTensor).to(device)

        batch_size = img_R1.size(0)

        img_R = torch.cat((img_R2, img_R1), dim=1)
        K = torch.cat((raw_K, raw_K), dim=0)

        norm_img_L2 = 0.004 * img_L2
        norm_img_R1 = 0.004 * img_R1
        norm_img_R2 = 0.004 * img_R2

        inv_depth_img_R2 = depth_net(img_R2)
        T_2to1, _ = odometry_net(img_R)

        T = torch.cat((T_R2L, T_2to1), dim=0)

        SE3 = generate_se3(T)
        inv_depth = torch.cat((inv_depth_img_R2, inv_depth_img_R2), dim=0)
        depth = (1 / (inv_depth + 1e-4))

        pts3D = geo_transform(depth, SE3, K)
        proj_coords = pin_hole_project(pts3D, K)

        Isrc = torch.cat((norm_img_L2, norm_img_R1), dim=0)
        warp_Itgt = inverse_warp(Isrc, proj_coords)

        warp_Itgt_LR = warp_Itgt[:batch_size, :, :, :]
        warp_Itgt_R12 = warp_Itgt[batch_size:, :, :, :]

        out_of_bound = 1 - (warp_Itgt_LR == 0).prod(1, keepdim=True).type_as(warp_Itgt_LR)
        diff_LR = (norm_img_R2 - warp_Itgt_LR) * out_of_bound
        LR_error = diff_LR.abs().mean()

        out_of_bound = 1 - (warp_Itgt_R12 == 0).prod(1, keepdim=True).type_as(warp_Itgt_R12)
        diff_R12 = (norm_img_R2 - warp_Itgt_R12) * out_of_bound
        R12_error = diff_R12.abs().mean()

        # reconstruction_error = photometric_reconstruction_loss(0.004*img_R2, 0.004*img_R1, 0.004*img_L2, depth, T_2to1, T_R2L, intrinsics, inv_intrinsics)
        smooth_error = smooth_loss(depth)

        loss = LR_error + R12_error + smooth_error

        total_loss += loss.item()
        lr_total += LR_error.item()
        r12_total += R12_error.item()
        smooth_total += smooth_error.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Train epoch {}: loss: {:.6f} LR-loss: {:.6f} R12-loss: {:.6f} smooth-loss: {:.6f}".format(epoch, 
        total_loss/len(train_loader), lr_total/len(train_loader), r12_total/len(train_loader), smooth_total/len(train_loader)))



@torch.no_grad()
def validate(model, depth_net, val_loader, epoch, output_dir, use_float=False):
    global device
    global best_val_loss
    if use_float:
        model.set_fix_method(nfp.FIX_NONE)
    else:
        model.set_fix_method(nfp.FIX_FIXED)
    model.eval()
    val_loss = 0
    for data, target in val_loader:
        data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
        output, _ = model(data)
        output = generate_se3(output)
        output = output.view(-1, 4, 4).type(torch.FloatTensor).to(device)
        val_loss += F.l1_loss(output, target).item()# sum up batch loss

    val_loss /= len(val_loader.dataset)
    is_best = val_loss < best_val_loss
    if not use_float:
        best_val_loss = val_loss if is_best else best_val_loss
    print('Test set: Average loss: {:.4f} [BEST:{}]'.format(val_loss, is_best if not use_float else 'N/A'))
    if use_float:
        return
    if is_best and args.gpu_id in range(2):
        torch.save(model.state_dict(), output_dir/"best_vo_checkpoint.pth.tar")
        torch.save(depth_net.state_dict(), output_dir/"best_depth_checkpoint.pth.tar")
    elif is_best:
        torch.save(model.module.state_dict(), output_dir/"best_vo_checkpoint.pth.tar")
        torch.save(depth_net.module.state_dict(), output_dir/"best_depth_checkpoint.pth.tar")


def main():
    global device
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
    vo_input_fix, vo_output_fix = False, False
    vo_conv_weight_fix = [False, False, False, False, False, False]
    #vo_conv_weight_fix = [True] * 6
    vo_fc_weight_fix = [False, False, False]
    vo_conv_output_fix = [False, False, False, False, False, False]
    #vo_conv_output_fix = [True] * 6
    vo_fc_output_fix = [False, False, False]
    odometry_net = FixOdometryNet(bit_width=BITWIDTH, input_fix=vo_input_fix, output_fix=vo_output_fix,
        conv_weight_fix=vo_conv_weight_fix, fc_weight_fix=vo_fc_weight_fix,
        conv_output_fix=vo_conv_output_fix, fc_output_fix=vo_fc_output_fix
    ).to(device)
#    odometry_net = OdometryNet().to(device)
    #depth_net = DepthNet().to(device)
    depth_net = DispNetS().to(device)

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

    cudnn.benchmark = True
    if args.cuda and args.gpu_id in range(2):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    elif args.cuda:
        odometry_net = torch.nn.DataParallel(odometry_net)
        depth_net = torch.nn.DataParallel(depth_net)

    optim_params = [
        {'params': odometry_net.parameters(), 'lr': args.lr},
        {'params': depth_net.parameters(), 'lr': args.lr}
    ]

    # model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer = optim.Adam(optim_params, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    print("=> validating before training")
    #validate(odometry_net, depth_net, val_loader, 0, output_dir, True)
    print("=> training & validating")
    #validate(odometry_net, depth_net, val_loader, 0, output_dir)
    for epoch in range(1, args.epochs+1):
        train(odometry_net, depth_net, train_loader, epoch, optimizer)
        validate(odometry_net, depth_net, val_loader, epoch, output_dir)

    #odometry_net.print_fix_configs()

if __name__ == '__main__':
    main()
    
    

