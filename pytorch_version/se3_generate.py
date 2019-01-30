import torch
import torch.nn as nn
import numpy as np
from torch.autograd.function import once_differentiable


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
                #print(R[i,0].shape)
                #print(uw_x[i,0].shape)
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

generate_se3 = SE3_Generator_KITTI.apply
