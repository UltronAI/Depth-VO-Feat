import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_

def conv_relu(in_channels, out_channels, kernel_size=3, padding=1, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.ReLU(inplace=True)
    )

class FeatExtractor(nn.Module):

    def __init__(self):
        super(FeatExtractor, self).__init__()

        self.conv_1_b5 = conv_relu(3, 32, kernel_size=3, padding=1, stride=1)
        self.conv_2_b5 = conv_relu(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv_3_b5 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)

        self.conv_1_b4 = conv_relu(35, 32, kernel_size=3, padding=1, stride=2)
        self.conv_2_b4 = conv_relu(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv_3_b4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)

        self.conv_1_b3 = conv_relu(35, 32, kernel_size=3, padding=1, stride=2)
        self.conv_2_b3 = conv_relu(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv_3_b3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)

        self.conv_1_b2 = conv_relu(35, 32, kernel_size=3, padding=1, stride=2)
        self.conv_2_b2 = conv_relu(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv_3_b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)

        self.conv_1_b1 = conv_relu(35, 32, kernel_size=3, padding=1, stride=2)
        self.conv_2_b1 = conv_relu(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv_3_b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)

        self.conv_3_b1_up = nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2, groups=32)
        self.conv_3_bb2_up = nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2, groups=32)
        self.conv_3_bb3_up = nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2, groups=32)
        self.conv_3_bb4_up = nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2, groups=32)

    def forward(self, imgs):
        imgs_b4 = F.interpolate(imgs, scale_factor=0.5, mode="bilinear")
        imgs_b3 = F.interpolate(imgs_b4, scale_factor=0.5, mode="bilinear")
        imgs_b2 = F.interpolate(imgs_b3, scale_factor=0.5, mode="bilinear")

        conv_1_b5 = self.conv_1_b5(imgs)
        conv_2_b5 = self.conv_2_b5(conv_1_b5)
        conv_3_b5 = self.conv_3_b5(conv_2_b5)
        im_conv_3_b5_concat = torch.cat((imgs, conv_3_b5), dim=1)

        conv_1_b4 = self.conv_1_b4(im_conv_3_b5_concat)
        conv_2_b4 = self.conv_2_b4(conv_1_b4)
        conv_3_b4 = self.conv_3_b4(conv_2_b4)
        im_conv_3_b4_concat = torch.cat((imgs_b4, conv_3_b4), dim=1)

        conv_1_b3 = self.conv_1_b3(im_conv_3_b4_concat)
        conv_2_b3 = self.conv_2_b3(conv_1_b3)
        conv_3_b3 = self.conv_3_b3(conv_2_b3)
        im_conv_3_b3_concat = torch.cat((imgs_b3, conv_3_b3), dim=1)

        conv_1_b2 = self.conv_1_b2(im_conv_3_b3_concat)
        conv_2_b2 = self.conv_2_b2(conv_1_b2)
        conv_3_b2 = self.conv_3_b2(conv_2_b2)
        im_conv_3_b2_concat = torch.cat((imgs_b2, conv_3_b2), dim=1)

        conv_1_b1 = self.conv_1_b1(im_conv_3_b2_concat)
        conv_2_b1 = self.conv_2_b1(conv_1_b1)
        conv_3_b1 = self.conv_3_b1(conv_2_b1)

        conv_3_b1_up = self.conv_3_b1_up(conv_3_b1)
        conv_3_bb2 = conv_3_b2 + conv_3_b1_up

        conv_3_bb2_up = self.conv_3_bb2_up(conv_3_bb2)
        conv_3_bb3 = conv_3_b3 + conv_3_bb2_up

        conv_3_bb3_up = self.conv_3_bb3_up(conv_3_bb3)
        conv_3_bb4 = conv_3_b4 + conv_3_bb3_up

        conv_3_bb4_up = self.conv_3_bb4_up(conv_3_bb4)
        pred_feat = conv_3_b5 + conv_3_bb4_up
        return pred_feat

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)
        
