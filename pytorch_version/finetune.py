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
from se3_generate import *
import cv2, os
import numpy as np
from path import Path

from tqdm import tqdm
import nics_fix_pt as nfp
import argparse
import pose_transforms
from dataset import pose_framework_KITTI

# hyper-parameters
BITWIDTH = 8

# argparser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--checkpoint', default=None, type=str)
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

def train(model, train_loader, epoch, optimizer):
    global device
    model.set_fix_method(nfp.FIX_AUTO)
    model.train()
    criterion = OdometryLoss().to(device)
    total_loss = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc='Train epoch %d' % epoch,
            leave=False, ncols=80):
        data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        output = generate_se3(output)
        output = output.view(-1, 4, 4).type(torch.FloatTensor).to(device)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        #if batch_idx % args.log_interval == 0:
        #    print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()), end="")
        #print("")
    print("Train epoch {}: mean loss = {:.6f}".format(epoch, total_loss/len(train_loader)))

@torch.no_grad()
def validate(model, val_loader, epoch, output_dir, use_float=False):
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
    print('Test set: Average loss: {:.6f} [BEST:{}]'.format(val_loss, is_best if not use_float else 'N/A'))
    if use_float:
        return
    if is_best and args.gpu_id in range(4):
        torch.save(model.state_dict(), output_dir/"best_checkpoint.pth.tar")
    elif is_best:
        torch.save(model.module.state_dict(), output_dir/"best_checkpoint.pth.tar")

class OdometryLoss(nn.modules.Module):
    def __init__(self):
        super(OdometryLoss, self).__init__()
        self.gt_mean = np.array([ 9.9978113e-1,  3.1038835e-5, -1.3910837e-3, -1.5050563e-3,
                        -2.4995565e-5,  9.9999690e-1,  1.0897936e-4, -1.3767095e-2,
                         1.3919161e-3, -9.5542389e-5,  9.9978083e-1,  8.1921977e-1]).reshape((3, 4))
        self.gt_std = np.array([5.2705163e-4, 3.0792558e-3, 2.0978536e-2, 3.0138873e-2,
                       3.0692818e-3, 1.9443671e-5, 3.7066913e-3, 9.9259941e-3,
                       2.0979958e-2, 3.6987357e-3, 5.2486762e-4, 2.7169874e-1]).reshape((3, 4))

    def forward(self, pred, gt):
        batch_size = pred.size(0)
        mean = torch.from_numpy(self.gt_mean).view(-1, 3, 4).expand_as(pred[:, :3, :]).type_as(pred)
        std = torch.from_numpy(self.gt_std).view(-1, 3, 4).expand_as(pred[:, :3, :]).type_as(pred)
        pred_ = (pred[:, :3, :]-mean)/std
        gt_ = (gt[:, :3, :]-mean)/std
        loss = F.mse_loss(pred_, gt_)
        return loss 

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
    train_set = pose_framework_KITTI(
        dataset_dir, args.train_sequences, 
        transform=train_transform,
        seed=args.seed
    )
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
    input_fix, output_fix = False, False
    conv_weight_fix = [True, True, True, True, True, True]
    #conv_weight_fix = [True] * 6
    fc_weight_fix = [False, False, False]
    conv_output_fix = [True, True, True, True, True, True]
    #conv_output_fix = [True] * 6
    fc_output_fix = [False, False, False]
    model = FixOdometryNet(bit_width=BITWIDTH, input_fix=input_fix, output_fix=output_fix,
        conv_weight_fix=conv_weight_fix, fc_weight_fix=fc_weight_fix,
        conv_output_fix=conv_output_fix, fc_output_fix=fc_output_fix
    ).to(device)

    # init weights of model
    if args.checkpoint is None:
        model.init_weights()
    elif args.checkpoint:
        weights = torch.load(args.checkpoint)
        model.load_state_dict(weights)

    cudnn.benchmark = True
    if args.cuda and args.gpu_id in range(4):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    elif args.cuda:
        model = torch.nn.DataParallel(model)

    # model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    #print("=> validating before training")
    #validate(model, val_loader, 0, output_dir, True)
    print("=> training & validating")
    #validate(model, val_loader, 0, output_dir)
    for epoch in range(1, args.epochs+1):
        train(model, train_loader, epoch, optimizer)
        validate(model, val_loader, epoch, output_dir)

    model.print_fix_configs()

if __name__ == '__main__':
    main()
    
    

