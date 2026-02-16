'''
python visualize_track_4d.py
'''

import numpy as np

import viser
import open3d as o3d
from tqdm import tqdm
import viser.transforms as tf
from viser.extras import ViserUrdf
import time, os
from pathlib import Path
from utils.dc_utils import read_video_frames, save_video
from utils.util import get_intrinsics_from_file
import matplotlib.pyplot as plt
import argparse
import json

def colored_depths_to_pc(depths, frames, fx, fy, cx, cy, downsample_factor: int = -1):
    
    width, height = depths[0].shape[-1], depths[0].shape[-2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - cx) / fx
    y = (y - cy) / fy

    pcds = []

    for i, (color_image, depth) in enumerate(zip(frames, depths)):
        z = np.array(depth)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(color_image).reshape(-1, 3) / 255.0

        if downsample_factor > 1:
            # randomly downsample points and colors
            indices = np.random.choice(points.shape[0], points.shape[0] // downsample_factor, replace=False)
            points = points[indices]
            colors = colors[indices]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcds.append(pcd)

    return pcds

def tracks_to_pc(pred_tracks, pred_visibility, depths, fx, fy, cx, cy):

    pcds = []
    num_frames, num_points, _ = pred_tracks.shape

    for i in range(num_frames):
        points = pred_tracks[i]  # (N, 2)
        visibility = pred_visibility[i]  # (N,)
        depth = depths[i] # this is depth map of shape (H, W)

        # Assign rainbow colors to each point
        colors = plt.get_cmap("rainbow")(np.linspace(0, 1, num_points))[:, :3]  # (N, 3)

        # Convert 2D points to 3D using depth
        try:
            z = depth[points[:, 1].astype(int), points[:, 0].astype(int)]
        except Exception as e:
            # some points are out of image bounds
            out_of_bounds_mask = (
                (points[:, 0] < 0) | (points[:, 0] >= depth.shape[1]) |
                (points[:, 1] < 0) | (points[:, 1] >= depth.shape[0])
            )
            points[out_of_bounds_mask] = 0
            z = depth[points[:, 1].astype(int), points[:, 0].astype(int)]
            z[out_of_bounds_mask] = 0.0  # set out of bounds depth to 0.0
        x = (points[:, 0] - cx) / fx * z
        y = (points[:, 1] - cy) / fy * z
        points_3d = np.stack((x, y, z), axis=-1)

        # Mask out invisible points
        points_3d[visibility < 0.5] = np.array([0.0, 0.0, 0.0])
        colors[visibility < 0.5] = np.array([0.0, 0.0, 0.0])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcds.append(pcd)

    return pcds

def load_camera_pose(pose_path: Path):
    with open(pose_path, "r") as f:
        pose_data = json.load(f)
    rotation = np.array(pose_data["camera_rotation_matrix_world"], dtype=np.float64)
    translation = np.array(pose_data["camera_translation_world"], dtype=np.float64)
    return rotation, translation

def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray):
    translation = translation.reshape(1, 3)
    return (points @ rotation.T) + translation

def apply_transform_to_pcds(pcds, rotation: np.ndarray, translation: np.ndarray):
    for pcd in pcds:
        points = np.asarray(pcd.points)
        world_points = transform_points(points, rotation, translation)
        pcd.points = o3d.utility.Vector3dVector(world_points)

def outlier_inlier_mask(points: np.ndarray, z_thresh: float = 3.5, min_points: int = 10) -> np.ndarray:
    if points.shape[0] < min_points:
        return np.ones(points.shape[0], dtype=bool)
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median, axis=1)
    median_distance = np.median(distances)
    mad = np.median(np.abs(distances - median_distance))
    if mad == 0.0:
        return np.ones(points.shape[0], dtype=bool)
    robust_z = 0.6745 * (distances - median_distance) / mad
    return np.abs(robust_z) <= z_thresh

def filter_false_keypoints(tracked_pcds):
        '''
        Filter out false keypoints due to depth errors.
        These keypoints' z value is typically too low or x value is too behind.
        '''
        for pcd in tracked_pcds:
            points_3d_world = np.asarray(pcd.points)
            valid_mask = np.any(points_3d_world != 0.0, axis=1)
            valid_points = points_3d_world[valid_mask]
            if valid_points.size > 0:
                inlier_mask = outlier_inlier_mask(valid_points)
                outlier_mask = valid_mask.copy()
                outlier_mask[valid_mask] = ~inlier_mask
                points_3d_world[outlier_mask] = 0.0
            # NOTE: Some hardcode for our robot setup
            false_mask = (points_3d_world[:, 2] < 0.602) | (points_3d_world[:, 0] < -0.5) | (np.abs(points_3d_world[:, 1]) > 0.5)
            points_3d_world[false_mask] = 0.0
            pcd.points = o3d.utility.Vector3dVector(points_3d_world)
        return tracked_pcds

def filter_pcds_max_distance(pcds, max_distance: float = 1.0):
    new_pcds = []
    for pcd in pcds:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if points.size == 0:
            continue
        distances = np.linalg.norm(points, axis=1)
        keep_mask = distances <= max_distance
        points = points[keep_mask]
        colors = colors[keep_mask]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        new_pcds.append(pcd)
    return new_pcds

def save_keypoints_world(tracked_pcds, pred_visibility, save_path: Path):
    keypoints_world = []
    for i, pcd in enumerate(tracked_pcds):
        points = np.asarray(pcd.points)
        visibility = pred_visibility[i]
        points[visibility < 0.5] = np.array([0.0, 0.0, 0.0])
        keypoints_world.append(points)
    keypoints_world = np.array(keypoints_world)  # (T, N, 3)
    np.save(save_path, keypoints_world)

def rotation_matrix_to_wxyz(rotation: np.ndarray):
    m = rotation
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)

def get_num_frames(video_path: Path):
    frames, _ = read_video_frames(video_path, -1)
    return frames.shape[0]

def load_data(
    video_path: str,
    depth_path: Path,
    track_path: Path,
):
    frames, _ = read_video_frames(video_path, -1)
    depths = np.load(depth_path)['depths']
    tracks_data = np.load(track_path)
    pred_tracks = tracks_data['pred_tracks']  # (T, N, 2)
    pred_visibility = tracks_data['pred_visibility']  # (T, N)

    return frames, depths, pred_tracks, pred_visibility

def main(args):
    data_path = args.data_path
    exp_name = args.exp_name
    share = args.share
    rgb_stride = max(1, args.image_stride)
    is_load_pose = False

    track_size = 4

    # Camera intrinsics
    instrinsics_path = f"outputs/realsense/{exp_name}/intrinsics.npz"
    fx, fy, cx, cy = get_intrinsics_from_file(instrinsics_path if os.path.exists(instrinsics_path) else None)

    # get all paths
    print("Loading frames!")
    frames, depths, pred_tracks, pred_visibility = load_data(
        video_path=str(data_path / "video_gen" / exp_name / "gen_video_resized.mp4"),
        depth_path=str(data_path / "depth_pred" / exp_name / "depths.npz"),
        # video_path=str(data_path / "realsense" / exp_name / "color_video.mp4"),
        # depth_path=str(data_path / "realsense" / exp_name / "depth_video.npz"),
        track_path=str(data_path / "track" / exp_name / "pred_tracks.npz"),
    )
    num_frames = frames.shape[0]
    image_height, image_width = frames.shape[1:3]
    camera_fov = 2 * np.arctan2(image_height / 2.0, fx)
    camera_aspect = image_width / image_height
    default_pose_path = data_path / "realsense" / exp_name / "calibration" / "camera_pose.json"
    pose_path = args.pose_path if args.pose_path is not None else default_pose_path
    pose_path = Path(pose_path)
    if not pose_path.exists():
        print(f"Camera pose file not found: {pose_path}")
        # is_load_pose = True
        # camera_rotation_world = np.eye(3)
        # camera_translation_world = np.array([0.0, 0.0, 0.0])
        # print("Using default camera pose at origin.")
    else:
        is_load_pose = True
        camera_rotation_world, camera_translation_world = load_camera_pose(pose_path)
        print(f"Loaded camera pose from {pose_path}")

    pcds = colored_depths_to_pc(depths[:num_frames], frames[:num_frames], fx=fx, fy=fy, cx=cx, cy=cy, downsample_factor=1)
    # pcds = filter_pcds_max_distance(pcds, max_distance=1.0)
    
    track_pcds = tracks_to_pc(pred_tracks[:num_frames], pred_visibility[:num_frames], depths[:num_frames], fx=fx, fy=fy, cx=cx, cy=cy)
    
    if is_load_pose:
        apply_transform_to_pcds(pcds, camera_rotation_world, camera_translation_world)
        apply_transform_to_pcds(track_pcds, camera_rotation_world, camera_translation_world)
        filter_false_keypoints(track_pcds)
        save_keypoints_world(track_pcds, pred_visibility, Path(f"outputs/track/{exp_name}/keypoints_world.npy"))

    server = viser.ViserServer()
    if share:
        server.request_share_url(verbose=True)

    # set z axis
    if is_load_pose:
        server.scene.set_up_direction('+z')
    else:
        server.scene.set_up_direction('-z')

    # Initial camera pose.
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # client.camera.position = (-1.554, -1.013, 1.142)
        # client.camera.look_at = (-0.005, 2.283, -0.156)
        pass

    if is_load_pose:
        camera_quat = rotation_matrix_to_wxyz(camera_rotation_world)
        server.scene.add_frame(
            "/camera_pose",
            wxyz=tuple(camera_quat.tolist()),
            position=tuple(camera_translation_world.tolist()),
            axes_length=0.025,
            axes_radius=0.001,
            show_axes=True,
        )
        initial_camera_image = frames[0][::rgb_stride, ::rgb_stride]
        camera_frustum = server.scene.add_camera_frustum(
            "/camera_pose/frustum",
            fov=camera_fov,
            aspect=camera_aspect,
            scale=args.image_frustum_scale,
            wxyz=tf.SO3.identity().wxyz,
            position=(0.0, 0.0, 0.0),
            image=initial_camera_image,
        )
        server.scene.add_frame(
            "/camera_pose/frustum/axes",
            axes_length=0.05,
            axes_radius=0.005,
            show_axes=False,
        )

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.0001,
            max=0.01,
            step=1e-4,
            initial_value=0.001,
        )
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=30
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )
    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value
    
    if is_load_pose:
        camera_image_timestep = -1

        def update_camera_image(timestep: int):
            nonlocal camera_image_timestep
            camera_frustum.image = frames[timestep][::rgb_stride, ::rgb_stride]
            camera_image_timestep = timestep

        update_camera_image(gui_timestep.value)

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value % num_frames
        with server.atomic():
            # Toggle visibility.
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        if is_load_pose:
            update_camera_image(current_timestep)
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz if is_load_pose else tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    point_nodes: list[viser.PointCloudHandle] = []
    track_nodes: list[viser.PointCloudHandle] = []

    for i in tqdm(range(num_frames)):
        pcd = pcds[i]

        # Extract points and colors
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        else:
            # If no colors in PLY, create default colors based on height
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            normalized_z = (points[:, 2] - z_min) / (z_max - z_min)
            colors = np.zeros((len(points), 3), dtype=np.uint8)
            colors[:, 0] = (normalized_z * 255).astype(np.uint8)  # Red channel
            colors[:, 2] = ((1 - normalized_z) * 255).astype(np.uint8)  # Blue channel

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=points,
                colors=colors,
                point_size=gui_point_size.value,
            )
        )
        # Add track points
        track_pcd = track_pcds[i]
        track_points = np.asarray(track_pcd.points)
        track_colors = (np.asarray(track_pcd.colors) * 255).astype(np.uint8)
        track_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/track_point_cloud",
                points=track_points,
                colors=track_colors,
                point_size=gui_point_size.value * track_size,
                point_shape="rounded"
            )
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    
    if args.export:
        # Create serializer.
        # Set the initial camera pose.
        server.initial_camera.position = (-0.0, -0.0, -0.389)
        server.initial_camera.look_at = (0, 0, 0)
        server.initial_camera.up = (-0.0, 0.0, 1.0)
        serializer = server.get_scene_serializer()
    
    while True:
        # Update the timestep if we're playing.
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        # Update point size of both this timestep and the next one! There's
        # redundancy here, but this will be optimized out internally by viser.
        #
        # We update the point size for the next timestep so that it will be
        # immediately available when we toggle the visibility.
        point_nodes[gui_timestep.value].point_size = gui_point_size.value
        track_nodes[gui_timestep.value].point_size = gui_point_size.value * track_size
        point_nodes[
            (gui_timestep.value + 1) % num_frames
        ].point_size = gui_point_size.value
        track_nodes[
            (gui_timestep.value + 1) % num_frames
        ].point_size = gui_point_size.value * track_size
        if is_load_pose and camera_image_timestep != gui_timestep.value:
            update_camera_image(gui_timestep.value)

        if not args.export:
            # Sleep to control framerate.
            time.sleep(1.0 / gui_framerate.value)
        else:
            serializer.insert_sleep(1.0 / gui_framerate.value)
            if gui_timestep.value == num_frames - 1:
                data = serializer.serialize()
                os.makedirs("outputs/viser", exist_ok=True)
                Path(f"outputs/viser/{args.exp_name}.viser").write_bytes(data)
                exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default="outputs/", help='Path to the data directory')
    parser.add_argument('--exp_name', type=str, required=True, help='Timestamp string for the video')
    parser.add_argument('--share', action='store_true', help='Share the visualization online')
    parser.add_argument('--pose_path', type=Path, default=None, help='Optional path to a midpoint_pose.json file. Defaults to video_dex_deploy/calibration/<exp_name>/midpoint_pose.json')
    parser.add_argument('--image_stride', type=int, default=4, help='Stride for subsampling pixels when rendering the camera image')
    parser.add_argument('--image_frustum_scale', type=float, default=0.05, help='Scale for the rendered camera frustum')
    parser.add_argument('--export', action='store_true', help='Whether to export the visualization as a .viser file')
    args = parser.parse_args()
    main(args)
