# Dex4D-Vision

This is the codebase for the vision toolkit of the Dex4D project.


https://github.com/user-attachments/assets/e5e3569b-c00c-4eee-b70a-62351525210e


## Installation

Please follow the steps below to perform the installation:

### 1. Create virtual environment

```bash
conda create -n dex4d-vision python==3.11.0
conda activate dex4d-vision
```

### 2. Video Depth Anything

```bash
pip install -r requirements.txt
```

Download the checkpoints and put them under the `checkpoints` directory.
```bash
bash get_weights.sh
```

### 3. CoTracker

```bash
pip install 'imageio[ffmpeg]'
cd <PATH_FOR_COTRACKER>/
git clone git@github.com:facebookresearch/co-tracker.git
cd co-tracker/ && pip install -e .
```

### 4. RealSense

```bash
pip install pyrealsense2
```

And follow instructions [here](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md) to install the SDK.

### 5. SAM2

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

The following instructions will guide you the whole process of **first RGBD frame capture -> video generation -> video depth estimation -> offline point tracking**, as well as **real-time online point tracking**. You can also run each module separately.

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
python -m cotracker3 --filename outputs/video_gen/$exp_name/gen_video_resized.mp4 --grid_size 100 # --use-mask
```


### Visualize the point tracks in 4D

```bash
python visualize_track_4d.py --data_path outputs/ --exp_name $exp_name # --share
```

After these steps, you should be able to see:


https://github.com/user-attachments/assets/9774d441-87df-4bfb-a7c4-e1dd72fa7ff8



### Real-Time Online Point Tracking (with Apriltag Calibration)

```bash
python real_time_tracking.py --exp_name $exp_name
```

The real-time tracking process should look like this:



https://github.com/user-attachments/assets/cfb6c933-9160-44d9-a8bb-e6bd510cd67a




## Acknowledgement

This project is built upon the following open-source projects:

- [DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)
- [StoneT2000/simple-easyhec](https://github.com/StoneT2000/simple-easyhec)

We thank the authors for their open-source contributions.
