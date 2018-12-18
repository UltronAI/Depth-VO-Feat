import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_uniform_, zeros_
from torch.autograd.function import once_differentiable

def conv(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )

def fc(in_length, out_length):
    return nn.Sequential(
        nn.Linear(in_length, out_length),
        nn.ReLU(inplace=True)
    )

class SE3_Generator_KITTI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.batch_size = input.size(0)
        ctx.threshold = 1e-12

        # Define skew matrix of so3, . size = (batch_size, 1, 3, 3)
        uw = input[:, :3].cpu().numpy().copy()
        
        uw_x = np.zeros((ctx.batch_size, 1, 3, 3))
        uw_x[:, 0, 0, 1] = -uw[:, 2, 0, 0]
        uw_x[:, 0, 0, 2] = uw[:, 1, 0, 0]
        uw_x[:, 0, 1, 0] = uw[:, 2, 0, 0]
        uw_x[:, 0, 1, 2] = -uw[:, 0, 0, 0]
        uw_x[:, 0, 2, 0] = -uw[:, 1, 0, 0]
        uw_x[:, 0, 2, 1] = uw[:, 0, 0, 0]

        # Get translation lie algebra
        ut = input[:, 3:].cpu().numpy()
        ut = np.reshape(ut, (ctx.batch_size, 1, 3 ,1))

        # Calculate SO3 and T, i.e. rotation matrix (batchsize,1,3,3) and translation matrix (batchsize,1,1,3)
        R = np.zeros((ctx.batch_size, 1, 3, 3))
        R[:, 0] = np.eye(3)
        theta = np.linalg.norm(uw, axis=1) # theta.size = (batch_size, 1)
        for i in range(ctx.batch_size):
            if theta[i] ** 2 < ctx.threshold:
                R[i, 0] += uw_x[i, 0]
                continue
            else:
                c1 = np.sin(theta[i]) / theta
                c2 = 2 * np.sin(theta[i]/2) ** 2 / theta[i] ** 2
                c3 = ((theta[i] - np.sin(theta[i])) / theta[i] ** 3) ** 2
                print(R[i,0].shape)
                print(uw_x[i,0].shape)
                R[i, 0] = R[i, 0] + c1 * uw_x[i, 0] + c2 * np.dot(uw_x[i, 0], uw_x[i, 0])

        output = np.zeros((ctx.batch_size, 1, 4, 4))
        output[:, :, :3, :3] = R
        output[:, :, :3, 3] = np.matmul(R, ut)[:, :, :, 0]
        output[:, :, 3, 3] = 1

        #ctx.save_for_backward(uw, uw_x, ut, R, theta)
        ctx.uw, ctx.uw_x, ctx.ut, ctx.R, ctx.theta = uw, uw_x, ut, R, theta

        return torch.from_numpy(output).cuda()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        uw, uw_x, ut, R, theta = ctx.uw, ctx.uw_x, ctx.ut, ctx.R, ctx.theta
        batch_size, threshold = ctx.batch_size, ctx.threshold
        grad_input = np.zeros((batch_size, 6, 1, 1))

        # grad_output.diff .shape is (batch_size,1,4,4)
        dLdT = grad_output[:, :, :3, 3].cpu().numpy().copy()
        dLdT = dLdT[:, np.newaxis]

        # Rt implementation for dLdut is dLdT x R
        dLdut = np.matmul(dLdT, R)
        grad_input[:, 3:, 0, 0] = dLdut[:, 0, 0]
        # Gradient correction for dLdR. '.' R also affect T, need update dLdR
        grad_corr = np.matmul(np.swapaxes(dLdT, 2, 3), np.swapaxes(ut, 2, 3))  # from (b,hw,4,1) to (b,4,hw,1)

        # dLduw
        dLdR = grad_output[:, :, :3, :3].cpu().numpy().copy()
        dLdR = dLdR + grad_corr
        dLduw = np.zeros((batch_size, 3))
        generators = np.zeros((3,3,3))
        generators[0] = np.array([[0,0,0],[0,0,1],[0,-1,0]])
        generators[1] = np.array([[0,0,-1],[0,0,0],[1,0,0]])
        generators[2] = np.array([[0,1,0],[-1,0,0],[0,0,0]])

        for index in range(3):
            I3 = np.zeros((batch_size, 1, 3, 3))
            I3[:, 0] = np.eye(3)
            ei = np.zeros((batch_size, 1, 3, 1))
            ei[:, 0, index] = 1
            cross_term = np.matmul(uw_x, np.matmul(I3-R, ei))
            cross = np.zeros((batch_size, 1, 3, 3))
            cross[:, 0, 0, 1] = -cross_term[:, 0, 2, 0]
            cross[:, 0, 0, 2] = cross_term[:, 0, 1, 0]
            cross[:, 0, 1, 0] = cross_term[:, 0, 2, 0]
            cross[:,0,1,2] = -cross_term[:,0,0,0]
            cross[:,0,2,0] = -cross_term[:,0,1,0]
            cross[:,0,2,1] = cross_term[:,0,0,0]

            dRduw_i = np.zeros((batch_size, 1, 3, 3))
            for j in range(batch_size):
                if theta[j] ** 2 < threshold:
                    dRduw_i[j] = generators[index]
                else:
                    dRduw_i[j, 0] = np.matmul((uw[j,index]*uw_x[j,0] + cross[j,0])/(theta[j]**2), R[j,0])
            dLduw[:, index] = np.sum(np.sum(dLdR*dRduw_i, axis=2), axis=2)[:, 0]

        grad_input[:, :3, 0, 0] = dLduw
        return torch.from_numpy(grad_input).type(torch.FloatTensor).cuda()

class OdometryNet(nn.Module):
    
    def __init__(self):
        super(OdometryNet, self).__init__()
        self.img_width = 608
        self.img_height = 160
        
        conv_channels = [16, 32, 64, 128, 256, 256]
        self.conv1 = conv(6,                conv_channels[0], kernel_size=7)
        self.conv2 = conv(conv_channels[0], conv_channels[1], kernel_size=5)
        self.conv3 = conv(conv_channels[1], conv_channels[2])
        self.conv4 = conv(conv_channels[2], conv_channels[3])
        self.conv5 = conv(conv_channels[3], conv_channels[4])
        self.conv6 = conv(conv_channels[4], conv_channels[5])

        self.fc1 = fc(conv_channels[5] * 3 * 10, 512)
        self.fc2 = fc(512,              512)

        self.fc_pose = nn.Linear(512, 6)

    def forward(self, x):
        assert (x.size(1) == 6 and x.size(2) == 160 and x.size(3) == 608), \
            print("input format is invalid.")

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)

        in_fc1 = out_conv6.view(out_conv6.size(0), -1)
        out_fc1 = self.fc1(in_fc1)
        out_fc2 = self.fc2(out_fc1)
        temporal_pose = self.fc_pose(out_fc2)
        temporal_pose = temporal_pose.view(temporal_pose.size(0), temporal_pose.size(1), 1, -1)
        se3 = SE3_Generator_KITTI.apply(temporal_pose)
        return se3

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

