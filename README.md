# Dex4D-Vision


## Installation

### Video Depth Anything

```bash
pip install -r requirements.txt
```

Download the checkpoints and put them under the `checkpoints` directory.
```bash
bash get_weights.sh
```

### CoTracker

```bash
pip install 'imageio[ffmpeg]'
cd <PATH_FOR_COTRACKER>/
git clone git@github.com:facebookresearch/co-tracker.git
cd co-tracker/ && pip install -e .
```

### RealSense

```bash
pip install pyrealsense2
```

And follow instructions [here](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md) to install the SDK.

### SAM2

```bash
cd <PATH_FOR_SAM2>/
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
# Download checkpoints
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

## Usage

The following instructions will guide you the whole process of **first RGBD frame capture -> video generation -> video depth estimation -> offline point tracking**, as well as **online point tracking**. You can also run each module separately.

### RGBD Frame Capture

First, specify the experiment name:

```bash
export exp_name="YOUR_EXP_NAME"
```

Then run the realsense capture script:

```bash
python realsense.py --exp_name $exp_name
```

You shall find the captured RGBD frames in `outputs/realsense/$exp_name/`.

### Video Generation

After first frame capture, use your favorite video generation model to generate a video and put it in `outputs/video_gen/$exp_name/<YOUR_VIDEO_NAME>.mp4`. You can also resize the video to the same size as the original RGB frame using `resize_video.py`:

```bash
python resize_video.py --video_path outputs/video_gen/$exp_name/<YOUR_VIDEO_NAME>.mp4
```

### Video Depth Estimation

```bash
python3 run.py --input_video outputs/video_gen/$exp_name/gen_video_resized.mp4 --encoder vitl --save_npz # --metric
```

For more options, please check [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) for details.

### Offline Point Tracking

```bash
CUDA_VISIBLE_DEVICES=1 python -m cotracker3 --filename outputs/video_gen/$exp_name/gen_video_resized.mp4 --grid_size 100 # --use-mask
```


### Visualize the point tracks in 4D

```bash
python visualize_track_4d.py --data_path outputs/ --exp_name $exp_name # --share
```


### Online Real-Time Point Tracking (with Apriltag Calibration)

```bash
python real_time_tracking.py --exp_name $exp_name
```


## Acknowledgement

This project is built upon the following open-source projects:

- [DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)
- [StoneT2000/simple-easyhec](https://github.com/StoneT2000/simple-easyhec)

We thank the authors for their open-source contributions.