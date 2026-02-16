from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import trimesh
import tyro
import time
import argparse
import matplotlib.cm as cm

class RealSenseCamera():
    def __init__(self, width=640, height=480, fps=30, exp_name=None, save_intrinsics=True):
        print("=> Initializing RealSense camera...")
        
        # Configure streams
        self.width = width
        self.height = height
        self.fps = fps

        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 1)
        self.spatial.set_option(rs.option.filter_smooth_delta, 50)
        self.spatial.set_option(rs.option.holes_fill, 3)
        self.hole_filling = rs.hole_filling_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.thr_filter = rs.threshold_filter()
        self.thr_filter.set_option(rs.option.min_distance,0.01)
        self.thr_filter.set_option(rs.option.max_distance,16.0)


        # 注册数据流，并对其图像
        self.align = rs.align(rs.stream.color)
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        rs_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        ### d415
        #
        rs_config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
        # rs_config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)

        # check相机是不是进来了
        connect_device = []
        for d in rs.context().devices:
            print('Found device: ',
                d.get_info(rs.camera_info.name), ' ',
                d.get_info(rs.camera_info.serial_number))
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                connect_device.append(d.get_info(rs.camera_info.serial_number))
        
        # 确认相机并获取相机的内部参数
        self.pipeline = rs.pipeline()
        rs_config.enable_device(connect_device[0])
        # pipeline_profile1 = pipeline1.start(rs_config)
        profile = self.pipeline.start(rs_config)
        color_stream = profile.get_stream(rs.stream.color)

        ### Fetch Intrinsics ###
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.intrinsic = np.array(
            [
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        print(f"Camera Intrinsics:\n {repr(self.intrinsic)}")

        # Save the color image and intrinsics
        if exp_name is not None:
            self.exp_name = exp_name
        else:
            self.exp_name = time.strftime("%Y%m%d-%H%M%S")

        if save_intrinsics:
            self.output_dir = Path("outputs/realsense") / self.exp_name
            self.output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.output_dir / "intrinsics.npz", intrinsic=self.intrinsic)
            print(f"Saved intrinsics to {self.output_dir}")

        print("Warming up the camera...")
        skip_frames = 60
        for _ in range(skip_frames):
            self.get_image()

    
    def depth_filter(self, depth_frame):
        # depth_frame = self.decimation.process(depth_frame)
        depth_frame = self.thr_filter.process(depth_frame)
        depth_frame = self.depth_to_disparity.process(depth_frame)
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.disparity_to_depth.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)
        return depth_frame
    
    def get_image(self, enable_depth=True):
        '''
        Return type: np.ndarray, color_image: H x W x 3, depth_image: H x W (in meters)
        '''
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        cframe = frames.get_color_frame()
        if not cframe:
            print("No frame")
        color_image = np.asanyarray(cframe.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        if enable_depth:
            dframe = frames.get_depth_frame()
            if not dframe:
                print("No depth frame")
            dframe = self.depth_filter(dframe)
            depth_image = np.asanyarray(dframe.get_data()) / 1000.0  # convert to meters
        else:
            depth_image = None

        return color_image, depth_image


    def capture_image(self, enable_depth=True):
        print("=> Capturing image...")
        ### Fetch one color image ###
        color_image, depth_image = self.get_image(enable_depth=enable_depth)

        # Save the color image
        cv2.imwrite(str(self.output_dir / "color_image.png"), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        print(f"Saved color image to {self.output_dir}")
        if enable_depth:
            # Save the raw depth image
            np.savez_compressed(self.output_dir / "depth_image.npz", depth=depth_image)
            print(f"Saved raw depth image to {self.output_dir}")
            # Visualize the depth image
            colormap = np.array(cm.get_cmap("inferno").colors)
            depth_vis = 1 / (depth_image + 1e-6)  # visualize inverse depth
            d_min, d_max = max(depth_vis.min(), 0), min(depth_vis.max(), 5)
            depth_vis = ((depth_vis - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_vis] * 255).astype(np.uint8)
            cv2.imwrite(str(self.output_dir / "depth_image.png"), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
            print(f"Saved depth visualization to {self.output_dir}")

    def capture_video(self, duration=10, enable_depth=True):
        print("=> Capturing video...")
        ### Capture a video sequence ###
        total_frames = duration * self.fps

        print(f"Capturing video for {duration} seconds...")
        video_frames = []
        depth_frames = []
        for i in range(total_frames):
            color_image, depth_image = self.get_image(enable_depth=enable_depth)
            video_frames.append(color_image)
            depth_frames.append(depth_image)
            if (i + 1) % self.fps == 0:
                print(f"Captured {(i + 1) // self.fps} seconds of video...")

        # Save the video frames as a video file
        video_path = self.output_dir / "color_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = video_frames[0].shape
        video_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
        for frame in video_frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
        if enable_depth:
            # Save the raw depth frames as a npz file
            np.savez_compressed(self.output_dir / "depth_video.npz", depths=np.array(depth_frames))
            print(f"Saved raw depth video to {self.output_dir / 'depth_video.npz'}")
            # Visualize the depth frames as a video file
            depth_video_path = self.output_dir / "depth_video.mp4"
            depth_video_writer = cv2.VideoWriter(str(depth_video_path), fourcc, self.fps, (width, height))
            colormap = np.array(cm.get_cmap("inferno").colors)
            depth_vis_s = 1 / (np.array(depth_frames) + 1e-6)  # visualize inverse depth
            d_min, d_max = max(depth_vis_s.min(), 0), min(depth_vis_s.max(), 5)
            for depth_vis in depth_vis_s:
                depth_vis = ((depth_vis - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                depth_vis = (colormap[depth_vis] * 255).astype(np.uint8)
                depth_video_writer.write(cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
            depth_video_writer.release()
            print(f"Saved depth visualization to {depth_video_path}")
        print(f"Saved color video to {video_path}")

    def stop(self):
        # Stop the pipeline
        self.pipeline.stop()
        print("Camera stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=640, help='Width of the color image')
    parser.add_argument('--height', type=int, default=480, help='Height of the color image')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the color image')
    parser.add_argument('--exp_name', type=str, default=None, help='If specified, use this timestamp string for saving outputs')
    args = parser.parse_args()
    
    camera = RealSenseCamera(width=args.width, height=args.height, fps=args.fps, exp_name=args.exp_name)
    camera.capture_image(enable_depth=True)
    camera.capture_video(duration=5, enable_depth=True)
    camera.stop()