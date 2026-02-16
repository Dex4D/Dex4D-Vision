# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
CUDA_VISIBLE_DEVICES=1 python run.py --input_video ./assets/example_videos/sora2_hand.MP4 --output_dir ./outputs/depth/ --encoder vitl --metric
'''

import argparse
import numpy as np
import os
import torch
import re

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video
from utils.util import get_intrinsics_from_file
from visualize_track_4d import colored_depths_to_pc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--metric', action='store_true', help='use metric model')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
    # parser.add_argument('--focal-length-x', default=1000, type=float,
    #                     help='Focal length along the x-axis.')
    # parser.add_argument('--focal-length-y', default=1000, type=float,
    #                     help='Focal length along the y-axis.')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder], metric=args.metric)
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/{checkpoint_name}_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

    exp_name = args.input_video.split("/")[-2]
    output_dir = f"outputs/depth_pred/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    instrinsics_path = f"outputs/realsense/{exp_name}/intrinsics.npz"
    fx, fy, cx, cy = get_intrinsics_from_file(instrinsics_path if os.path.exists(instrinsics_path) else None)

    import open3d as o3d
    os.makedirs(os.path.join(output_dir, f'pcds'), exist_ok=True)
    if not args.metric:
        # relative depths are inverse depths
        depths = 1.0 / (depths + 1e-6)
    realsense_depth_path = f"outputs/realsense/{exp_name}/depth_image.npz"
    if os.path.exists(realsense_depth_path):
        realsense_depth = np.load(realsense_depth_path)['depth'] # (H, W)
        # scale the predicted depth videos from the realsense depth
        realsense_depth_median = np.median(realsense_depth)
        first_estimated_depth_median = np.median(depths[0])
        scale_factor = realsense_depth_median / first_estimated_depth_median
    else:
        # Use a default scaling factor
        if args.metric:
            scale_factor = 0.75
        else:
            scale_factor = 550.0
    
    print(f"Scaling factor: {scale_factor}")
    depths = depths * scale_factor

    pcds = colored_depths_to_pc(depths, frames, fx=fx, fy=fy, cx=cx, cy=cy)
    for i, pcd in enumerate(pcds):
        o3d.io.write_point_cloud(os.path.join(output_dir, f"pcds", 'point' + str(i).zfill(4) + '.ply'), pcd)

    processed_video_path = os.path.join(output_dir, 'src.mp4')
    depth_vis_path = os.path.join(output_dir, 'vis.mp4')
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

    if args.save_npz:
        depth_npz_path = os.path.join(output_dir, 'depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
    if args.save_exr:
        depth_exr_dir = os.path.join(output_dir, 'depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR
        import Imath
        for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()


