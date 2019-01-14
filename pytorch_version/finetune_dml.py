# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from fixmodel import FixOdometryNet
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
parser.add_argument('--float-checkpoint', default=None, type=str)
parser.add_argument('--fix-checkpoint', default=None, type=str)
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                    help='SGD momentum (default: 0.8)')
parser.add_argument('--weight-decay', type=float, default=1e-8, metavar='WD',
                    help='SGD weight decay (default: 1e-8)')
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 2019)')
parser.add_argument('-b', '--batch-size', default=64, type=int)

parser.add_argument('-g', '--gpu-id', type=int, metavar='N', default=-1)
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
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

def train(float_model, fix_model, train_loader, epoch, optimizer):
    global device
    float_model.set_fix_method(nfp.FIX_NONE)
    float_model.train()
    fix_model.set_fix_method(nfp.FIX_AUTO)
    fix_model.train()
    total_loss_float = 0
    total_loss_fix = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc='Train epoch %d' % epoch,
            leave=False, ncols=80):
        data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()

        float_output = float_model(data).view(-1, 4, 4).type(torch.FloatTensor).to(device)
        loss_1 = F.smooth_l1_loss(float_output, target)
        total_loss_float += loss_1.item()

        fix_output = fix_model(data).view(-1, 4, 4).type(torch.FloatTensor).to(device)
        loss_2 = F.smooth_l1_loss(fix_output, target)
        total_loss_fix += loss_2.item()

        loss_3 = F.kl_div(fix_output, float_output)

        loss = loss_1 + loss_2 + loss_3
        loss.backward()
        optimizer.step()
    print("Train epoch {}: mean loss (float) = {:.6f} | mean loss (fix) = {:.6f}".format(epoch, total_loss_float/len(train_loader), total_loss_fix/len(train_loader)))

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
        output = model(data).view(-1, 4, 4).type(torch.FloatTensor).to(device)
        val_loss += F.l1_loss(output, target).item()# sum up batch loss

    val_loss /= len(val_loader.dataset)
    is_best = val_loss < best_val_loss
    if not use_float:
        best_val_loss = val_loss if is_best else best_val_loss
    print('Test set: Average loss: {:.4f} [{}][BEST:{}]'.format(val_loss, 'float' if use_float else 'fix', is_best if not use_float else 'N/A'))
    if use_float:
        return
    if is_best and args.gpu_id in range(2):
        torch.save(model.state_dict(), output_dir/"best_checkpoint.pth.tar")
    elif is_best:
        torch.save(model.module.state_dict(), output_dir/"best_checkpoint.pth.tar")


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
    float_model = FixOdometryNet(bit_width=BITWIDTH).to(device)

    input_fix, output_fix = False, False
    #conv_weight_fix = [False, False, False, False, False, False]
    conv_weight_fix = [True] * 6
    fc_weight_fix = [False, False, False]
    #conv_output_fix = [False, False, False, False, False, False]
    conv_output_fix = [True] * 6
    fc_output_fix = [False, False, False]
    fix_model = FixOdometryNet(bit_width=BITWIDTH, input_fix=input_fix, output_fix=output_fix,
        conv_weight_fix=conv_weight_fix, fc_weight_fix=fc_weight_fix,
        conv_output_fix=conv_output_fix, fc_output_fix=fc_output_fix
    ).to(device)

    # init weights of model
    if args.float_checkpoint is None:
        float_model.init_weights()
    else:
        float_weights = torch.load(args.float_checkpoint)
        float_model.load_state_dict(float_weights)

    if args.fix_checkpoint is None:
        fix_model.init_weights()
    else:
        fix_weights = torch.load(args.float_checkpoint)
        fix_model.load_state_dict(fix_weights)

    cudnn.benchmark = True
    if args.cuda and args.gpu_id in range(2):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    elif args.cuda:
        float_model = torch.nn.DataParallel(float_model)
        fix_model = torch.nn.DataParallel(fix_model)

    # model = model.to(device)
    optim_params = [
        {'params': float_model.parameters(), 'lr': args.lr},
        {'params': fix_model.parameters(), 'lr': args.lr}
    ]
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer = optim.Adam(optim_params, betas=(args.momentum, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    print("=> validating before training")
    validate(float_model, val_loader, 0, output_dir, True)
    validate(fix_model, val_loader, 0, output_dir)
    print("=> training & validating")
    for epoch in range(1, args.epochs+1):
        train(float_model, fix_model, train_loader, epoch, optimizer)
        validate(float_model, val_loader, 0, output_dir, True)
        validate(fix_model, val_loader, epoch, output_dir)

    fix_model.print_fix_configs()

if __name__ == '__main__':
    main()
    
    

