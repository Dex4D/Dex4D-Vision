import torch
import numpy as np
from realsense import RealSenseCamera
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor
from cotracker.models.core.model_utils import smart_cat, get_points_on_a_grid
from cotracker3 import CoTrackerOnlineSparsePredictor
from segmentation import build_simple_sam2_predictor, InteractiveSegmentation
import argparse
import time
import os
import imageio
import matplotlib.cm as cm
import cv2
from matplotlib import pyplot as plt
from two_april_tag_calibration import calibrate
from three_april_tag_calibration import calibrate_camera_pose_from_third_tag

def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False):
    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    if is_depths:
        frames = 1 / (frames + 1e-6)  # visualize inverse depth
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = max(frames.min(), 0), min(frames.max(), 5)
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            writer.append_data(depth_vis)
    else:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])

    writer.close()

def stream_video_tracking(args):
    print("Starting real-time video stream with tracking...")
    exp_name = args.exp_name if args.exp_name is not None else time.strftime("%Y%m%d-%H%M%S")
    device = torch.device(args.device)
    grid_size = args.grid_size

    # prepare camera and visualizer
    try:
        calibrate(
            tag_size=0.06,
            tag_ids=None,
            save_dir=f"outputs/realsense/{exp_name}/calibration"
        )
        # calibrate_camera_pose_from_third_tag(
        #     tag_size=0.06,
        #     third_tag_dir='calibration/third_tag',
        #     save_dir=f"outputs/realsense/{exp_name}/calibration"
        # )
    except Exception as e:
        print(f"April tag calibration failed: {e}. Proceeding without calibration.")
    camera = RealSenseCamera(width=args.width, height=args.height, fps=args.fps, exp_name=exp_name)
    vis = Visualizer(
        save_dir=f"outputs/track/{exp_name}", 
        pad_value=0, 
        linewidth=1, 
        fps=30,
        # mode='cool',
        tracks_leave_trace=-1
    )

    # get initial frame for mask generation
    first_frame, _ = camera.get_image(enable_depth=False)  # H x W x 3
    interactive_segmentation = InteractiveSegmentation()
    print("==> SAM2 successfully loaded!")
    masks = interactive_segmentation.get_segmentation([first_frame]) # [1, H, W]
    mask = masks[0]

    # save the initial mask in numpy and image format
    os.makedirs(f"outputs/track/{exp_name}", exist_ok=True)
    np.savez_compressed(os.path.join(f"outputs/track/{exp_name}", "initial_mask.npz"), mask=mask)
    mask_color = np.array([30, 144, 255], dtype=np.float32)
    mask_overlay = mask.astype(np.float32)[..., None] * mask_color.reshape(1, 1, -1)
    masked_vis = mask_overlay * 0.6 + first_frame * 0.4
    masked_vis[mask == 0] = first_frame[mask == 0]
    masked_vis = masked_vis.astype(np.uint8)
    cv2.imwrite(
        os.path.join(f"outputs/track/{exp_name}", "initial_mask.png"),
        cv2.cvtColor(masked_vis, cv2.COLOR_RGB2BGR),
    )
    print(f"=> Initial mask saved to outputs/track/{exp_name}")

    # Run Online CoTracker:
    cotracker: CoTrackerOnlinePredictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
    cotracker = CoTrackerOnlineSparsePredictor(online_tracker=cotracker).to(device)
    cotracker.eval()
    window_frames = []
    depth_frames = []
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

    i = 0
    # max_history = cotracker.step * 2 + 2
    while True:
        try:
            start_time = time.time()

            # Get the next frame from the camera
            frame, depth = camera.get_image(enable_depth=True)
            window_frames.append(frame)
            depth_frames.append(depth)
            
            # if len(window_frames) > max_history:
            #     window_frames = window_frames[-max_history:]
            #     depth_frames = depth_frames[-max_history:]

            if i % cotracker.step == 0 and i > 0:
                pred_tracks, pred_visibility = _process_step(window_frames, is_first_step, grid_size) # (1, T, N, 2), (1, T, N), while T is growing
                if pred_visibility is not None:
                    curr_visible_point_num = pred_visibility[0, -1].sum().item()
                else:
                    curr_visible_point_num = 0

                print(
                    f"Processing frame {i} in {time.time() - start_time:.6f} seconds, {curr_visible_point_num} points visible",
                    end="\r",
                )
                
                # real time visualization
                if not is_first_step and args.real_time_vis:
                    colors = plt.get_cmap("rainbow")(np.linspace(0, 1, pred_tracks.shape[2]))[:, :3]  # (N, 3)
                    track_points = pred_tracks[0, -1, :, :].cpu() # (N, 2)
                    track_visibility = pred_visibility[0, -1, :].cpu() # (N,)
                    track_points = track_points[track_visibility > 0.5]  # only show visible points
                    colors = colors[track_visibility > 0.5]
                    
                    # Draw circles instead of single pixels
                    frame_viz = frame.copy()
                    for point, color in zip(track_points, colors):
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(frame_viz, (x, y), 5, (color * 255).astype(int).tolist(), -1)
                    
                    cv2.imshow("Real-Time Tracking", cv2.cvtColor(frame_viz, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                is_first_step = False
            
            i += 1

        except KeyboardInterrupt:
            print("Stopping video stream...")
            break
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % cotracker.step) - cotracker.step - 1:],
        is_first_step=is_first_step,
        grid_size=grid_size,
        grid_query_frame=0,
    )
    if len(window_frames) > pred_tracks.shape[1]:
        window_frames = window_frames[:pred_tracks.shape[1]]

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
        os.path.join(f"outputs/track/{exp_name}", "pred_tracks_online.npz"), 
        pred_tracks=pred_tracks[0].cpu().numpy(), # T N 2
        pred_visibility=pred_visibility[0].cpu().numpy() # T N
    )

    # save window_frames and depth_frames as video
    realsense_dir = f"outputs/realsense/{exp_name}"
    os.makedirs(realsense_dir, exist_ok=True)
    if window_frames:
        cv2.imwrite(
            os.path.join(realsense_dir, "color_image.png"),
            cv2.cvtColor(window_frames[0], cv2.COLOR_RGB2BGR),
        )
    if depth_frames and depth_frames[0] is not None:
        np.savez_compressed(
            os.path.join(realsense_dir, "depth_image.npz"), depth=depth_frames[0]
        )
        colormap = np.array(cm.get_cmap("inferno").colors)
        depth_vis = 1 / (depth_frames[0] + 1e-6)  # visualize inverse depth
        d_min, d_max = max(depth_vis.min(), 0), min(depth_vis.max(), 5)
        depth_vis = ((depth_vis - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        depth_vis = (colormap[depth_vis] * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(realsense_dir, "depth_image.png"),
            cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR),
        )
    color_video_path = f"outputs/realsense/{exp_name}/color_video.mp4"
    depth_video_path = f"outputs/realsense/{exp_name}/depth_video.mp4"
    depth_npz_path = f"outputs/realsense/{exp_name}/depth_video.npz"
    save_video(np.array(window_frames), color_video_path, fps=args.fps)
    save_video(np.array(depth_frames), depth_video_path, fps=args.fps, is_depths=True)
    np.savez_compressed(depth_npz_path, depths=np.array(depth_frames))
    print(f"Realsense data saved to outputs/realsense/{exp_name}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=640, help='Width of the color image')
    parser.add_argument('--height', type=int, default=480, help='Height of the color image')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for the color image')
    parser.add_argument('--grid_size', type=int, default=60, help='Grid size for CoTracker')
    parser.add_argument('--exp_name', type=str, default=None, help='If specified, use this timestamp string for saving outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--real_time_vis', action='store_true', help='Whether to show real-time visualization during tracking')
    args = parser.parse_args()

    stream_video_tracking(args)
