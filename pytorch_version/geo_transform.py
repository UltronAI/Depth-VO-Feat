import torch
import threading
import time
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

    x_matrix = torch.from_numpy(np.tile(np.tile(np.arange(width).reshape((1, width)), [height, 1]), [num, 1, 1])).type_as(depthmap)
    y_matrix = torch.from_numpy(np.tile(np.tile(np.arange(height).reshape((height, 1)), [1, width]), [num, 1, 1])).type_as(depthmap)

    fx = torch.from_numpy(np.tile(np.tile(cam_intrinsic[:, 0, 0, 0].cpu().numpy().reshape((num, 1)), [1, height]).reshape(num, height, 1), [1, 1, width])).type_as(depthmap)
    fy = torch.from_numpy(np.tile(np.tile(cam_intrinsic[:, 1, 0, 0].cpu().numpy().reshape((num, 1)), [1, height]).reshape(num, height, 1), [1, 1, width])).type_as(depthmap)
    cx = torch.from_numpy(np.tile(np.tile(cam_intrinsic[:, 2, 0, 0].cpu().numpy().reshape((num, 1)), [1, height]).reshape(num, height, 1), [1, 1, width])).type_as(depthmap)
    cy = torch.from_numpy(np.tile(np.tile(cam_intrinsic[:, 3, 0, 0].cpu().numpy().reshape((num, 1)), [1, height]).reshape(num, height, 1), [1, 1, width])).type_as(depthmap)

    # fx, fy, cx, cy
    x = (x_matrix - cx) / fx * depthmap[:, 0, :, :]
    y = (y_matrix - cy) / fy * depthmap[:, 0, :, :]

    transformed_points[:, 0, :, :] = pose[:, 0, 0, 0].view(-1, 1, 1).type_as(depthmap) * x + pose[:, 0, 0, 1].view(-1, 1, 1).type_as(depthmap) * y + \
                                    pose[:, 0, 0, 2].view(-1, 1, 1).type_as(depthmap) * depthmap[:, 0, :, :]# + pose[:, 0, 0, 3].type_as(depthmap)
    exit(0)
    transformed_points[:, 1, :, :] = pose[:, 0, 1, 0] * x + pose[:, 0, 1, 1] * y + \
                                    pose[:, 0, 1, 2] * depthmap[:, 0, :, :] + pose[:, 0, 1, 3]
    transformed_points[:, 2, :, :] = pose[:, 0, 2, 0] * x + pose[:, 0, 2, 1] * y + \
                                    pose[:, 0, 2, 2] * depthmap[:, 0, :, :] + pose[:, 0, 2, 3]

    exit(0)
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

lock = threading.Lock()

class myThread (threading.Thread):
    def __init__(self, img, proj_coords, warp_img, index):
        threading.Thread.__init__(self)
        self.img = img
        self.proj_coords = proj_coords
        self.warp_img = warp_img
        self.index = index
    def run(self):
        # print "Starting " + self.name
        lock.acquire()
        inverse_warp_unit(self.img, self.proj_coords, self.warp_img, self.index)
        lock.release()

def inverse_warp_unit(img, proj_coords, warp_img, index):
    num, channel, height, width = img.size()

    x = index % width
    y = (index / width) % height
    n = index / width / height

    xx = proj_coords[n, 0, y, x]
    yy = proj_coords[n, 1, y, x]

    x1 = torch.floor(xx)
    x2 = x1 + 1
    y1 = torch.floor(yy)
    y2 = y1 + 1

    wx2 = xx - torch.float(x1)
    wx1 = float(x2) - xx
    wy2 = yy - torch.float(y1)
    wy1 = float(y2) - yy

    for cc in range(channel):
        if (x1 >= 0 and x1 <= width-1 and y1 >= 0 and y1 <= height-1):
            warp_img[n, cc, y, x] += wx1 * wy1 * img[n, cc, y1, x1]
        if (x1 >= 0 and x1 <= width-1 and y2 >= 0 and y2 <= height-1):
            warp_img[n, cc, y, x] += wx1 * wy2 * img[n, cc, y2, x1]
        if (x2 >= 0 and x2 <= width-1 and y1 >= 0 and y1 <= height-1):
            warp_img[n, cc, y, x] += wx2 * wy1 * img[n, cc, y1, x2]
        if (x2 >= 0 and x2 <= width-1 and y2 >= 0 and y2 <= height-1):
            warp_img[n, cc, y, x] += wx2 * wy2 * img[n, cc, y2, x2]

def inverse_warp(img, proj_coords):
    # bottom[0] --> Image (N,C,H,W)
    # bottom[1] --> projection coordinate (N,2,H,W)
    # top[0]    --> Warp iamge (N,C,H,W)
    num = img.size(0)
    channel = img.size(1)
    height = img.size(2)
    width = img.size(3)
    warp_img = torch.from_numpy(np.zeros((num, channel, height, width)))

    threads = []
 
    for i in range(num * height * width):
        thread_ = myThread(img, proj_coords, warp_img, i)
        thread_.start()
        threads.append(thread_)

    for t in threads:
        t.join()

    return warp_img
