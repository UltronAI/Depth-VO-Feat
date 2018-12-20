import torch
from network import OdometryNet
import cv2
import numpy as np

import nics_fix_pt as nfp

img_width = 608;
img_height = 160;

def getImage(img_path, transform=True): 
    # ----------------------------------------------------------------------
    # Get and preprocess image
    # ----------------------------------------------------------------------
    img = cv2.imread(img_path)
    if img is None:
        print("img_path: " + img_path)
        assert img is not None, "Image reading error. Check whether your image path is correct or not."
    img = cv2.resize(img, (img_width, img_height))
    img = img.transpose((2,0,1))
    img = img.astype(np.float32)
    if transform:
        img[0] -= 104
        img[1] -= 117
        img[2] -= 123
    return img

def main():
    img1_path = "/home/gaof/workspace/00/image_2/000000.png"
    img2_path = "/home/gaof/workspace/00/image_2/000001.png"

    img1 = getImage(img1_path, True)
    img2 = getImage(img2_path, True)

    input_data = np.zeros((1, 6, img_height, img_width))
    input_data[:, :3, :, :] = img2
    input_data[:, 3:, :, :] = img1

    input_tensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()

    model = OdometryNet().cuda()
    model.set_fix_method(nfp.FIX_AUTO)
    model.init_weights()
    torch.save(model.state_dict(), 'begin_checkpoint.pth.tar')
    optim_params = [
        {'params': model.parameters(), 'lr':1e-4}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(0.9, 0.999),
                                 weight_decay=0.8)
    loss = torch.nn.L1Loss()
    tgt_temp = torch.from_numpy(np.zeros((1, 1, 4, 4))).type(torch.FloatTensor).cuda()
    model.train()
    model.print_fix_configs()
    optimizer.zero_grad()
    output = model(input_tensor)
    loss_ = loss(output, tgt_temp)
    #print(output.cpu().detach().numpy().shape)
    loss_.backward()
    optimizer.step()
    torch.save(model.state_dict(), 'after_checkpoint.pth.tar')
    model.print_fix_configs()

if __name__ == "__main__":
    main()
