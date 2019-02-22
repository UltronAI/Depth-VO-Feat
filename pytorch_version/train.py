import argparse
import time
import csv
import os

import numpy as np
import tqdm
import torch
from path import Path
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import custom_transforms
from PoseExpNet_sfm import PoseExpNet
from DispNetS import DispNetS
from utils import tensor2array, save_checkpoint, save_path_formatter
from inverse_warp import inverse_warp
from se3_generate import *
from dataset import pose_framework_KITTI
from un_dataset import dataset

from loss_functions_sfm import photometric_reconstruction_loss, explainability_loss, smooth_loss, compute_errors


parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset-dir", default='/home/share/kitti_odometry/dataset/', type=str, help="Dataset directory")
parser.add_argument("--train-sequences", default=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], 
                    type=str, nargs='*', help="sequences to train")
parser.add_argument("--test-sequences", default=['00'], type=str, nargs='*', help="sequences to test")
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                    help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('--smooth-loss-factor', type=float, help='scale factor used to compute smooth loss', metavar='W', default=2)
parser.add_argument('-g', '--gpu-id', type=int, metavar='N', default=-1)
parser.add_argument('--output-dir', type=str, default='./checkpoints')

n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    global n_iter, device
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.gpu_id == 0 or args.gpu_id == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    else:
        args.gpu_id = -1

    print("=> will save everthing to {}".format(args.output_dir))
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor()])#, normalize])

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

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    # the network for single-view depth prediction
    # disp_net = models.DispNetS().to(device)
    output_exp = args.mask_loss_weight > 0
    if not output_exp:
        print("=> no mask loss, PoseExpnet will only output pose")
    disp_net = DispNetS().to(device)
    pose_exp_net = PoseExpNet().to(device)

    if args.pretrained_exp_pose:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_exp_pose)
        pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        pose_exp_net.init_weights()

    if args.pretrained_disp:
        print("=> using pre-trained weights for Dispnet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        disp_net.init_weights()

    cudnn.benchmark = True
    if args.gpu_id < 0:
        disp_net = torch.nn.DataParallel(disp_net)
        pose_exp_net = torch.nn.DataParallel(pose_exp_net)

    print('=> setting adam solver')

    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_exp_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        # train for one epoch
        train(args, train_loader, disp_net, pose_exp_net, optimizer, args.epoch_size)
        validate(args, pose_exp_net, disp_net, val_loader, epoch, output_dir)

def train(args, train_loader, disp_net, pose_exp_net, optimizer, epoch_size):
    global n_iter, device
    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight
    scale_factor = args.smooth_loss_factor

    # switch to train mode
    disp_net.train()
    pose_exp_net.train()

    for batch_idx, (img_R1, img_L2, img_R2, intrinsics, intrinsics_inv, raw_K, T_R2L) in tqdm(enumerate(train_loader), desc='Train epoch %d' % epoch, leave=False, ncols=80):
        tgt_img = img_R2.to(device)
        ref_imgs = [img.to(device) for img in [img_R1, img_L2]]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        disparities = disp_net(tgt_img)
        depth = [1/disp for disp in disparities]
        explainability_mask, pose = pose_exp_net(tgt_img, ref_imgs)

        loss_1 = photometric_reconstruction_loss(tgt_img, ref_imgs,
                                                 intrinsics, intrinsics_inv,
                                                 depth, explainability_mask, pose,
                                                 args.rotation_mode, args.padding_mode)
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask)
        else:
            loss_2 = 0

        loss_3 = smooth_loss(depth, scale_factor)
        loss_4 = F.mse_loss(pose[:, 1], T_R2L)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + loss_4

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



best_val_loss = 99999

@torch.no_grad()
def validate(args, model, depth_net, val_loader, epoch, output_dir, use_float=False):
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
    elif is_best:
        torch.save(model.module.state_dict(), output_dir/"best_vo_checkpoint.pth.tar")
        torch.save(depth_net.module.state_dict(), output_dir/"best_depth_checkpoint.pth.tar")

if __name__ == '__main__':
    main()
