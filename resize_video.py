import os
import argparse
import numpy as np
import cv2
from utils.dc_utils import read_video_frames, save_video

parser = argparse.ArgumentParser(description='Resize Generated Video')
parser.add_argument('--width', type=int, default=640, help='Target width of the video')
parser.add_argument('--height', type=int, default=480, help='Target height of the video')
parser.add_argument('--video_path', type=str, required=True, help='Path to the generated video')
args = parser.parse_args()

width, height = args.width, args.height
video_dir = os.path.dirname(args.video_path)

video_frames, _ = read_video_frames(args.video_path, process_length=-1)
# resize video frames to original image size
resized_frames = []
for i in range(len(video_frames)):
    resized_frames.append(cv2.resize(video_frames[i], (width, height)))
resized_frames = np.array(resized_frames)
save_video(resized_frames, os.path.join(video_dir, "gen_video_resized.mp4"), fps=30)
print(f"Resized video saved to {os.path.join(video_dir, 'gen_video_resized.mp4')}")