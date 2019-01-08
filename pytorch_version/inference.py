# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from network import OdometryNet
import cv2, os
import numpy as np
from path import Path

import nics_fix_pt as nfp
import argparse
import pose_transforms
from load_00 import pose_framework_KITTI

# hyper-parameters
BITWIDTH = 8

# argparser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argument('--seed', type=int, default=2018, metavar='S',
                    help='random seed (default: 2018)')
parser.add_argument('-g', '--gpu-id', type=int, metavar='N', default=-1)
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
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

def save_result_poses(se3, output_dir, filename):
    f = open(os.path.join(output_dir, filename), 'a')
    
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

    f.writelines(line_to_write + "\n")
    f.close()

@torch.no_grad()
def inference(model, val_loader, output_dir):
    global device
    model.set_fix_method(nfp.FIX_NONE)
    model.eval()
    for _, (data, target) in enumerate(val_loader):
        data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
        output = model(data).view(-1, 4, 4).cpu().numpy()[0]
        #print(output.shape)
        #exit(0)
        save_result_poses(output, output_dir, 'pytorch_fix_pred.txt')
        #save_result_poses(target.cpu().numpy()[0], output_dir, 'gt.txt')

def main():
    global device
    print("=> will save everthing to {}".format(args.output_dir))
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    # Data loading code
    normalize = pose_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    valid_transform = pose_transforms.Compose([pose_transforms.ArrayToTensor()])

    print("=> fetching sequences in '{}'".format(args.dataset_dir))
    dataset_dir = Path(args.dataset_dir)
    print("=> preparing val set")
    val_set = pose_framework_KITTI(
        dataset_dir, "/home/gaof/workspace/Depth-VO-Feat/kitti_00.txt", 
        transform=valid_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    model = OdometryNet(bit_width=BITWIDTH).to(device)
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

    inference(model, val_loader, args.output_dir)

if __name__ == '__main__':
    main()
    
    

