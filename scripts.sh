export exp_name="bottlemug"


python realsense.py --exp_name $exp_name

python kling.py --image_path outputs/realsense/$exp_name/color_image.png --prompt "A hand grabs the bottle on the table and pours water to the mug, static camera view" --duration 5

python3 run.py --input_video outputs/video_gen/$exp_name/gen_video_resized.mp4 --encoder vitl --save_npz # --metric

CUDA_VISIBLE_DEVICES=1 python -m cotracker3 --filename outputs/video_gen/$exp_name/gen_video_resized.mp4 --grid_size 100 --use-mask

python visualize_track_4d.py --data_path outputs/ --exp_name $exp_name --share

# real time tracking
python real_time_tracking.py --exp_name $exp_name

python resize_video.py --video_path outputs/video_gen/0123-meat2bowl/Wan_Video_Generate_实景拍摄，男人轻轻抓住桌子上的肉，轻轻举起，然后放进碗里。人手不要遮挡肉，也不要有多余动作，使用静态相机视角。_.mp4