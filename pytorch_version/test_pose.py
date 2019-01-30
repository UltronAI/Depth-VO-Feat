import torch
from torch.autograd import Variable

from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from model import OdometryNet
from inverse_warp import pose_vec2mat


parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("--use-quant-model", action='store_true')

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

def SE3_cam2world(pred_poses):
    pred_SE3_world = []
    cur_t = np.eye(4)
    pred_SE3_world.append(cur_t)
    for pose in pred_poses:
        cur_t = np.dot(cur_t, pose)
        pred_SE3_world.append(cur_t)
    return pred_SE3_world

@torch.no_grad()
def main():
    args = parser.parse_args()
    from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework

    weights = torch.load(args.pretrained_posenet)
    pose_net = OdometryNet().to(device)
    pose_net.load_state_dict(weights)

    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']

        h,w,_ = imgs[0].shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]

        imgs = [np.transpose(img, (2,0,1)) for img in imgs]

        ref_imgs = []
        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            img = ((img/255 - 0.5)/0.5).to(device)
            if i == len(imgs)//2:
                tgt_img = img
            else:
                ref_imgs.append(img)

        _, poses = pose_net(tgt_img, ref_imgs)

        poses = poses.cpu()[0]
        poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])
        inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]

        if args.output_dir is not None:
            predictions_array[j] = final_poses

        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE','RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions_array)


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length


if __name__ == '__main__':
    main()