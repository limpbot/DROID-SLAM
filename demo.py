import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


import re
from pathlib import Path

def atoi(text):
    return int(text) if text.isdigit() else text

def sorted_image_list(image_list):
    return sorted(image_list, key=lambda f: [atoi(val) for val in re.split(r'(\d+)', Path(f).stem)])

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]
    image_list = sorted_image_list(image_list)

    print('first 10 image relative file paths')
    print(image_list[:10])

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def save_reconstruction(droid, reconstruction_path, imagedir, stride):

    from pathlib import Path
    import random
    import string

    # t = droid.video.counter.value
    #tstamps = droid.video.tstamp[:t].cpu().numpy()
    #images = droid.video.images[:t].cpu().numpy()
    #poses = droid.video.poses[:t].cpu().numpy()
    #intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    disps = droid.video.disps_up[:].cpu().numpy()

    image_list = sorted(os.listdir(imagedir))[::stride]
    image_list = sorted_image_list(image_list)

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    Path("reconstructions/{}/disps".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)

    for t, imfile in enumerate(image_list):
        # H x W
        # h1 = int(np.sqrt((384 * 512) / (H * W)) * H)
        # h1 = h1 - h1 % 8
        # w1 = int(np.sqrt((384 * 512) / (H * W)) * W)
        # w1 = w1 - w1 % 8
        image = cv2.imread(os.path.join(imagedir, imfile))
        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        h_pad = h1 - h1 % 8
        w_pad = w1 - w1 % 8
        disp = disps[t]
        disp = np.pad(disp, ((0, h_pad), (0, w_pad)))
        disp = cv2.resize(disp, (w0, h0))
        disp *= (w0 / w1)
        cv2.imwrite(f"reconstructions/{reconstruction_path}/depths/{Path(imfile).stem}.jpg", disp)

    # np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    # np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    # np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    # np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    # np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)
    #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path, imagedir=args.imagedir, stride=args.stride)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))

    print(type(traj_est))
    print(traj_est)

    from lietorch import SE3

    #torch.save(torch.from_numpy(traj_est), "reconstructions/{}/traj_est.pt".format(args.reconstruction_path))
    torch.save((SE3(torch.from_numpy(traj_est)).inv().matrix()), "reconstructions/{}/traj_est.pt".format(args.reconstruction_path))

    pts3d = []
    clr3d = []
    import open3d as o3d
    from visualization import droid_visualization
    import droid_backends

    droid_visualization_filter_thresh = 0.005

    with torch.no_grad():

        with droid.video.get_lock():
            t = droid.video.counter.value
            dirty_index, = torch.where(droid.video.dirty.clone())
            dirty_index = dirty_index

        droid.video.dirty[dirty_index] = False

        # convert poses to 4x4 matrix
        poses = torch.index_select(droid.video.poses, 0, dirty_index)
        disps = torch.index_select(droid.video.disps, 0, dirty_index)
        Ps = SE3(poses).inv().matrix().cpu().numpy()

        images = torch.index_select(droid.video.images, 0, dirty_index)
        images = images.cpu()[:, [2, 1, 0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
        points = droid_backends.iproj(SE3(poses).inv().data, disps, droid.video.intrinsics[0]).cpu()

        thresh = droid_visualization_filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

        count = droid_backends.depth_filter(
            droid.video.poses, droid.video.disps, droid.video.intrinsics[0], dirty_index, thresh)

        count = count.cpu()
        disps = disps.cpu()
        masks = ((count >= 2) & (disps > .5 * disps.mean(dim=[1, 2], keepdim=True)))

        for i in range(len(dirty_index)):
            pose = Ps[i]
            ix = dirty_index[i].item()

            mask = masks[i].reshape(-1)
            pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
            clr = images[i].reshape(-1, 3)[mask].cpu().numpy()

            pts3d.append(pts)
            clr3d.append(clr)
    ## add point actor ###
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.concatenate(pts3d))
    point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate(clr3d))

    o3d.io.write_point_cloud(filename="reconstructions/{}/pcl.ply".format(args.reconstruction_path), pointcloud=point_cloud)

