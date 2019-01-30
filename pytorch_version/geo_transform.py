import torch
import threading
import numpy as np

def geo_transform(depthmap, pose, cam_intrinsic):
    # bottom[0] --> DepthMap (N,1,H,W)
    # bottom[1] --> Transformation matrix (N,1,4,4)
    # bottom[2] --> camera intrinsic coefficient (N,4,1,1)
    # top[0]    --> transformed 3D points (N,3,H,W)
    num = depthmap.size(0)
    height = depthmap.size(2)
    width = depthmap.size(3)

    transformed_points = torch.from_numpy(np.zeros((num, 3, height, width)))

    x_matrix = torch.from_numpy(np.tile(np.tile(np.arange(width).reshape((1, width)), [height, 1]), [num, 1, 1]))
    y_matrix = torch.from_numpy(np.tile(np.tile(np.arange(height).reshape((height, 1)), [1, width]), [num, 1, 1]))

    # fx, fy, cx, cy
    x = (x_matrix - cam_intrinsic[:, 2, 0, 0]) / cam_intrinsic[:, 0, 0, 0] * depthmap[:, 0, :, :]
    y = (y_matrix - cam_intrinsic[:, 3, 0, 0]) / cam_intrinsic[:, 1, 0, 0] * depthmap[:, 0, :, :]

    transformed_points[:, 0, :, :] = pose[:, 0, 0, 0] * x + pose[:, 0, 0, 1] * y + \
                                    pose[:, 0, 0, 2] * depthmap[:, 0, :, :] + pose[:, 0, 0, 3]
    transformed_points[:, 1, :, :] = pose[:, 0, 1, 0] * x + pose[:, 0, 1, 1] * y + \
                                    pose[:, 0, 1, 2] * depthmap[:, 0, :, :] + pose[:, 0, 1, 3]
    transformed_points[:, 2, :, :] = pose[:, 0, 2, 0] * x + pose[:, 0, 2, 1] * y + \
                                    pose[:, 0, 2, 2] * depthmap[:, 0, :, :] + pose[:, 0, 2, 3]

    return transformed_points

def pin_hole_project(transformed_points, cam_intrinsic):
    # bottom[0] --> 3D points (N,C,H,W)
    # bottom[1] --> camera intrinsic (N,4,1,1)
    # top[0]    --> flows (N,2,H,W) [u,v]
    # top[1]    --> projection coordinat (N,2,H,W) [xp,yp]
    num = transformed_points.size(0)
    height = transformed_points.size(2)
    width = transformed_points.size(3)

    proj_coords = torch.from_numpy(np.zeros((num, 2, height, width)))

    x = transformed_points[:, 0, :, :]
    y = transformed_points[:, 1, :, :]
    z = transformed_points[:, 2, :, :]

    proj_coords[:, 0, :, :] = cam_intrinsic[:, 0, 0, 0] * x / (z + torch.tensor(1e-12)) + cam_intrinsic[:, 2, 0, 0]
    proj_coords[:, 1, :, :] = cam_intrinsic[:, 1, 0, 0] * y / (z + torch.tensor(1e-12)) + cam_intrinsic[:, 3, 0, 0]

    return proj_coords

def inverse_warp(img, proj_coords):
    # bottom[0] --> Image (N,C,H,W)
    # bottom[1] --> projection coordinate (N,K,H,W)
    # top[0]    --> Warp iamge (N,C,H,W)
    num = img.size(0)
    height = img.size(2)
    width = img.size(3)