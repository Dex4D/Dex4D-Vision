'''
The dexgrasp conda env does not support this script. Please use the easyhec conda env to run this script:
conda activate easyhec
CUDA_VISIBLE_DEVICES=1 python -m cotracker3

OR use subprocess to run this script in python code:
import subprocess
subprocess.run(f"CUDA_VISIBLE_DEVICES=1 conda run -n easyhec python -m cotracker3", shell=True)
'''


import torch
import numpy as np
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor
from cotracker.models.core.model_utils import smart_cat, get_points_on_a_grid
from torch.nn import functional as F
import os, sys
import argparse
import time

from segmentation import build_simple_sam2_predictor, InteractiveSegmentation


class CoTrackerOnlineSparsePredictor(torch.nn.Module):

    def __init__(self, online_tracker: CoTrackerOnlinePredictor = None):
        super().__init__()
        self.online_tracker = online_tracker
        self.step = online_tracker.step
        self.interp_shape = online_tracker.interp_shape

    @torch.no_grad()
    def set_masked_queries(
        self,
        video,
        segm_mask,
        grid_size,
        grid_query_frame=0,
    ):
        B, T, C, H, W = video.shape

        grid_pts = get_points_on_a_grid(
            grid_size, self.interp_shape, device=video.device
        )
        if segm_mask is not None:
            segm_mask = F.interpolate(
                segm_mask, tuple(self.interp_shape), mode="nearest"
            )
            point_mask = segm_mask[0, 0][
                (grid_pts[0, :, 1]).round().long().cpu(),
                (grid_pts[0, :, 0]).round().long().cpu(),
            ].bool()
            grid_pts = grid_pts[:, point_mask]

        queries = torch.cat(
            [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
            dim=2,
        ).repeat(B, 1, 1)

        self.online_tracker.queries = queries

    @torch.no_grad()
    def forward(
        self,
        video_chunk,
        is_first_step: bool = False,
        segm_mask: torch.Tensor = None,
        grid_size: int = 5,
        grid_query_frame: int = 0,
        add_support_grid=False,
    ):
        if is_first_step:
            self.online_tracker.model.init_video_online_processing()
            self.set_masked_queries(
                video_chunk,
                segm_mask=segm_mask,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
            )
            return (None, None)
        else:
            return self.online_tracker(
                video_chunk,
                is_first_step=is_first_step,
                queries=None,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
                add_support_grid=add_support_grid,
            )


def main(args):
    # set up video
    filename = args.filename
    grid_size = args.grid_size

    import imageio.v3 as iio
    frames = iio.imread(filename, plugin="FFMPEG")  # plugin="pyav"

    device = 'cuda'
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

    first_frame = video[0, 0] / 255.0  # C H W
    first_frame = first_frame.permute(1, 2, 0)[:, :, :3].cpu().numpy()  # H W C

    exp_name = filename.split("/")[-2]
    mask_path = args.mask_path
    if mask_path is None and args.use_mask:
        initial_mask_path = os.path.join("outputs/track", exp_name, "initial_mask.npz")
        if os.path.exists(initial_mask_path):
            mask_path = initial_mask_path

    if mask_path is not None:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        if mask_path.endswith(".npz"):
            mask = np.load(mask_path)["mask"]
        elif mask_path.endswith(".npy"):
            mask = np.load(mask_path)
        else:
            raise ValueError(f"Unsupported mask format: {mask_path}")
        print(f"=> Loaded existing mask from {mask_path}")
    else:
        interactive_segmentation = InteractiveSegmentation()
        print("==> SAM2 successfully loaded!")
        masks = interactive_segmentation.get_segmentation([first_frame]) # [1, H, W]
        mask = masks[0]

    # set up track visualizer
    os.makedirs(f"outputs/track/{exp_name}", exist_ok=True)
    vis = Visualizer(
        save_dir=f"outputs/track/{exp_name}", 
        pad_value=0, 
        linewidth=1, 
        fps=30,
        # mode='cool',
        # tracks_leave_trace=-1
    )

    if args.mode == 'offline':
        # Run Offline CoTracker:
        cotracker: CoTrackerPredictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        pred_tracks, pred_visibility = cotracker(
            video, 
            grid_size=grid_size,
            segm_mask=torch.from_numpy(mask)[None, None].to(device)
        ) # B T N 2,  B T N 1

        vis.visualize(video, pred_tracks, pred_visibility, filename="vis_offline")

    elif args.mode == 'online':
        # Run Online CoTracker:
        cotracker: CoTrackerOnlinePredictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
        cotracker = CoTrackerOnlineSparsePredictor(online_tracker=cotracker).to(device)
        cotracker.eval()
        window_frames = []
        is_first_step = True
        
        def _process_step(window_frames, is_first_step, grid_size, grid_query_frame=0):
            video_chunk = (
                torch.tensor(
                    np.stack(window_frames[-cotracker.step * 2:]), device=device
                ).float().permute(0, 3, 1, 2)[None]
            ) # 1, T, C, H, W
            pred_tracks, pred_visibility = cotracker(
                video_chunk,
                is_first_step=is_first_step,
                segm_mask=torch.from_numpy(mask)[None, None].to(device),
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
            )
            return pred_tracks, pred_visibility

        # Iterating over video frames, processing one window at a time:
        for i, frame in enumerate(
            iio.imiter(
                filename,
                plugin="FFMPEG",
            )
        ):  
            start_time = time.time()

            if i % cotracker.step == 0 and i != 0:
                pred_tracks, pred_visibility = _process_step(window_frames, is_first_step, grid_size)
                is_first_step = False
                print(
                    f"Processing frame {i}/{video.shape[1]} in {time.time() - start_time:.6f} seconds",
                    end="\r",
                )
                
            window_frames.append(frame)
        print(" " * 80, end="\r")
        
        pred_tracks, pred_visibility = _process_step(
            window_frames[-(i % cotracker.step) - cotracker.step - 1:],
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=0,
        )
        print("=> Tracks are computed. Visualizing...")

        # save a video with predicted tracks
        video = torch.tensor(np.stack(window_frames), device=device).permute(
            0, 3, 1, 2
        )[None]
        vis.visualize(
            video, pred_tracks, pred_visibility, query_frame=0, filename="vis_online"
        )

    # save predicted tracks and visibility
    np.savez(
        os.path.join(f"outputs/track/{exp_name}", "pred_tracks.npz"), 
        pred_tracks=pred_tracks[0].cpu().numpy(), # T N 2
        pred_visibility=pred_visibility[0].cpu().numpy() # T N
    )
    np.savez(
        os.path.join(f"outputs/track/{exp_name}", "pred_tracks_offline.npz"), 
        pred_tracks=pred_tracks[0].cpu().numpy(), # T N 2
        pred_visibility=pred_visibility[0].cpu().numpy() # T N
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='assets/example_videos/sora2_hand.MP4', help='path to the video file')
    parser.add_argument('--mode', type=str, default='offline', choices=['offline', 'online'], help='whether to use offline or online CoTracker')
    parser.add_argument('--grid_size', type=int, default=50, help='grid size for CoTracker')
    parser.add_argument('--use-mask', action='store_true', help='whether to use a provided mask for the first frame')
    parser.add_argument('--mask-path', type=str, default=None, help='path to a saved mask (.npz with key "mask" or .npy); skips interactive segmentation')
    args = parser.parse_args()
    main(args)
