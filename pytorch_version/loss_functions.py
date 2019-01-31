from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from inverse_warp import inverse_warp

def photometric_reconstruction_loss(img_R2, img_R1, img_L2, depth, T_2to1, T_R2L, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros'):
    reconstruction_loss = 0

    warped_R1 = inverse_warp(img_R1, depth, T_2to1, intrinsics, intrinsics_inv, rotation_mode, padding_mode)
    out_of_bound = 1 - (warped_R1 == 0).prod(1, keepdim=True).type_as(warped_R1)
    diff_R1 = (img_R2, warped_R1) * out_of_bound
    reconstruction_loss += diff_R1.abs().mean

    warped_L2 = inverse_warp(img_L2, depth, T_R2L, intrinsics, intrinsics_inv, rotation_mode, padding_mode)
    out_of_bound = 1 - (warped_L2 == 0).prod(1, keepdim=True).type_as(warped_L2)
    diff_L2 = (img_R2, warped_L2) * out_of_bound
    reconstruction_loss += diff_L2.abs().mean

    return reconstruction_loss


def smooth_loss(pred_map, scale_factor=1):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= scale_factor  # don't ask me why it works better
    return loss