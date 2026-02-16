'''
In this script, the april tag axes in image space are x-front (red), y-right (green), z-up (blue).
'''

import argparse
import json
import os
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

try:
    import pupil_apriltags as apriltag
    APRILTAG_AVAILABLE = True
    print("Using pupil-apriltags library")
except ImportError:
    APRILTAG_AVAILABLE = False
    print("Error: No AprilTag library found. Install with: pip install apriltag")
    exit(1)

try:
    cv2.setNumThreads(1)
except Exception:
    pass

from two_april_tag_calibration import (
    convert_pose_axes,
    get_camera_intrinsics,
    detect_apriltags,
    select_two_tags,
    compute_midpoint_pose,
    convert_pose_from_midpoint_to_world,
    invert_pose,
    log_pose,
    draw_tag_axes,
    draw_midpoint_axes,
)


def select_one_tag(detected_tags, expected_id=None):
    assert expected_id is not None, "expected_id must be provided"
    for tag in detected_tags:
        if tag["tag_id"] == expected_id:
            return tag, None
    return None, f"Missing expected tag ID {expected_id}"


def calibrate_third_tag_pose(
    tag_size,
    save_dir,
):
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    if not any(s.get_info(rs.camera_info.name) == "RGB Camera" for s in device.sensors):
        print("RealSense device with RGB sensor is required.")
        return

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    print("Starting RealSense pipeline...")
    pipeline.start(config)

    camera_matrix, dist_coeffs = get_camera_intrinsics(pipeline.get_active_profile())

    print("AprilTag Midpoint Mode:")
    print("  Press 'c' to capture and compute pose")
    print("  Press 'q' to quit")
    print("  Use --tag-ids if you must lock to specific IDs")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            preview = color_image.copy()
            cv2.putText(
                preview,
                "Press 'c' to capture midpoint pose | 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                f"Expecting tag size {tag_size} m",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Two-Tag Midpoint Pose", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting midpoint pose capture...")
                break
            if key != ord("c"):
                continue

            print("Capturing frame for AprilTag detection...")
            detections = detect_apriltags(color_image, camera_matrix, tag_size)

            tags_of_interest, reason = select_two_tags(detections, expected_ids=[0, 1])
            third_tag, reason_third = select_one_tag(detections, expected_id=2)
            if not tags_of_interest:
                print(f"No valid tag pair detected: {reason}")
                continue

            midpoint_rotation, midpoint_translation = compute_midpoint_pose(tags_of_interest)
            camera_rotation, camera_translation = invert_pose(
                midpoint_rotation, midpoint_translation 
            ) # camera pose in midpoint frame
            camera_rotation_to_third_tag, camera_translation_to_third_tag = invert_pose(
                third_tag["rotation_matrix"], third_tag["translation_vector"]
            ) # camera pose in third tag frame

            print(f"Detected tags {[tag['tag_id'] for tag in detections]}")
            for tag in detections:
                _, converted_tvec = convert_pose_axes(tag["rotation_matrix"], tag["translation_vector"])
                tvec = converted_tvec.reshape(-1)
                print(
                    f"  Tag {tag['tag_id']}: translation {tvec}, distance {tag['distance']:.3f} m"
                )

            midpoint_rotation_converted, midpoint_translation_converted = convert_pose_axes(
                midpoint_rotation, midpoint_translation
            )
            camera_rotation_converted, camera_translation_converted = convert_pose_axes(
                camera_rotation, camera_translation
            ) # camera pose in midpoint frame (axes converted)
            camera_rotation_to_third_tag, camera_translation_to_third_tag = convert_pose_axes(
                camera_rotation_to_third_tag, camera_translation_to_third_tag
            ) # camera pose in third tag frame (axes converted)
            third_tag_rotation_to_midpoint = (
                camera_rotation_converted @ camera_rotation_to_third_tag.T
            ) # third tag pose in midpoint frame (axes converted)
            third_tag_translation_to_midpoint = (
                camera_translation_converted
                - third_tag_rotation_to_midpoint @ camera_translation_to_third_tag
            ) # third tag pose in midpoint frame (axes converted)
            third_tag_rotation_to_world, third_tag_translation_to_world = convert_pose_from_midpoint_to_world(
                third_tag_rotation_to_midpoint, third_tag_translation_to_midpoint
            )

            log_pose(f"{'='*30}\nMidpoint pose in camera frame:", midpoint_rotation_converted, midpoint_translation_converted)
            log_pose(f"{'='*30}\nCamera pose in midpoint frame:", camera_rotation_converted, camera_translation_converted)
            log_pose(f"{'='*30}\nCamera pose in third tag frame:", camera_rotation_to_third_tag, camera_translation_to_third_tag)
            log_pose(f"{'='*30}\nThird tag pose in midpoint frame:", third_tag_rotation_to_midpoint, third_tag_translation_to_midpoint)
            log_pose(f"{'='*30}\nThird tag pose in world frame:", third_tag_rotation_to_world, third_tag_translation_to_world)

            annotated = color_image.copy()
            for tag in detections:
                draw_tag_axes(annotated, tag, camera_matrix, dist_coeffs)
            draw_midpoint_axes(annotated, midpoint_rotation, midpoint_translation, camera_matrix, dist_coeffs)

            cv2.namedWindow(
                "Midpoint Capture - Press 'c' to calibrate again, press 'q' to quit",
                cv2.WINDOW_AUTOSIZE,
            )
            cv2.imshow("Midpoint Capture - Press 'c' to calibrate again, press 'q' to quit", annotated)

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)

                original_path = os.path.join(save_dir, "color_frame.png")
                annotated_path = os.path.join(save_dir, "annotated_frame.png")
                cv2.imwrite(original_path, color_image)
                cv2.imwrite(annotated_path, annotated)

                selected_tags_summary = []
                for tag in tags_of_interest:
                    converted_rotation, converted_translation = convert_pose_axes(
                        tag["rotation_matrix"], tag["translation_vector"]
                    )
                    converted_rvec, _ = cv2.Rodrigues(converted_rotation)
                    selected_tags_summary.append(
                        {
                            "tag_id": tag["tag_id"],
                            "translation_vector": converted_translation.reshape(-1).tolist(),
                            "rotation_vector": converted_rvec.reshape(-1).tolist(),
                            "corners": tag["corners"],
                            "center": tag["center"],
                        }
                    )

                summary = {
                    "tag_size": tag_size,
                    "camera_matrix": camera_matrix.tolist(),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "selected_tags": selected_tags_summary,
                    "midpoint_rotation_matrix_camera": midpoint_rotation_converted.tolist(),
                    "midpoint_translation_camera": midpoint_translation_converted.reshape(-1).tolist(),
                    "camera_rotation_matrix_midpoint": camera_rotation_converted.tolist(),
                    "camera_translation_midpoint": camera_translation_converted.reshape(-1).tolist(),
                    "third_tag_rotation_matrix_midpoint": third_tag_rotation_to_midpoint.tolist(),
                    "third_tag_translation_midpoint": third_tag_translation_to_midpoint.reshape(-1).tolist(),
                    "third_tag_rotation_matrix_world": third_tag_rotation_to_world.tolist(),
                    "third_tag_translation_world": third_tag_translation_to_world.reshape(-1).tolist(),
                }
                with open(os.path.join(save_dir, "third_tag_pose.json"), "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"Saved capture data to {save_dir}, press 'c' to capture again or 'q' to quit.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")


def calibrate_camera_pose_from_third_tag(
    tag_size,
    third_tag_dir,
    save_dir,
):
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    if not any(s.get_info(rs.camera_info.name) == "RGB Camera" for s in device.sensors):
        print("RealSense device with RGB sensor is required.")
        return

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    print("Starting RealSense pipeline...")
    pipeline.start(config)

    camera_matrix, dist_coeffs = get_camera_intrinsics(pipeline.get_active_profile())

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            preview = color_image.copy()
            cv2.putText(
                preview,
                "Press 'c' to capture midpoint pose | 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                f"Expecting tag size {tag_size} m",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Two-Tag Midpoint Pose", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting midpoint pose capture...")
                break
            if key != ord("c"):
                continue

            print("Capturing frame for AprilTag detection...")
            detections = detect_apriltags(color_image, camera_matrix, tag_size)

            third_tag, reason_third = select_one_tag(detections, expected_id=2)

            if third_tag is None:
                print(f"No valid third tag detected: {reason_third}")
                continue

            # Read from saved calibration
            with open(os.path.join(third_tag_dir, "third_tag_pose.json"), "r") as f:
                saved_data = json.load(f)
            third_tag_rotation_to_world = np.array(saved_data["third_tag_rotation_matrix_world"])
            third_tag_translation_to_world = np.array(saved_data["third_tag_translation_world"]).reshape(3, 1)
            camera_rotation_to_third_tag, camera_translation_to_third_tag = invert_pose(
                third_tag["rotation_matrix"], third_tag["translation_vector"]
            ) # camera pose in third tag frame
            camera_rotation_to_third_tag, camera_translation_to_third_tag = convert_pose_axes(
                camera_rotation_to_third_tag, camera_translation_to_third_tag
            ) # camera pose in third tag frame (axes converted)
            camera_rotation_to_world = third_tag_rotation_to_world @ camera_rotation_to_third_tag
            camera_translation_to_world = (
                third_tag_rotation_to_world @ camera_translation_to_third_tag
                + third_tag_translation_to_world
            )
            log_pose(f"{'='*30}\nCamera pose in world frame:", camera_rotation_to_world, camera_translation_to_world)

            annotated = color_image.copy()
            for tag in detections:
                draw_tag_axes(annotated, tag, camera_matrix, dist_coeffs)

            cv2.namedWindow(
                "Midpoint Capture - Press 'c' to calibrate again, press 'q' to quit",
                cv2.WINDOW_AUTOSIZE,
            )
            cv2.imshow("Midpoint Capture - Press 'c' to calibrate again, press 'q' to quit", annotated)

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)

                original_path = os.path.join(save_dir, "color_frame.png")
                annotated_path = os.path.join(save_dir, "annotated_frame.png")
                cv2.imwrite(original_path, color_image)
                cv2.imwrite(annotated_path, annotated)

                selected_tags_summary = []
                for tag in detections:
                    converted_rotation, converted_translation = convert_pose_axes(
                        tag["rotation_matrix"], tag["translation_vector"]
                    )
                    converted_rvec, _ = cv2.Rodrigues(converted_rotation)
                    selected_tags_summary.append(
                        {
                            "tag_id": tag["tag_id"],
                            "translation_vector": converted_translation.reshape(-1).tolist(),
                            "rotation_vector": converted_rvec.reshape(-1).tolist(),
                            "corners": tag["corners"],
                            "center": tag["center"],
                        }
                    )

                summary = {
                    "tag_size": tag_size,
                    "camera_matrix": camera_matrix.tolist(),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "selected_tags": selected_tags_summary,
                    "third_tag_rotation_matrix_world": third_tag_rotation_to_world.tolist(),
                    "third_tag_translation_world": third_tag_translation_to_world.reshape(-1).tolist(),
                    "camera_rotation_matrix_world": camera_rotation_to_world.tolist(),
                    "camera_translation_world": camera_translation_to_world.reshape(-1).tolist(),
                }
                with open(os.path.join(save_dir, "camera_pose.json"), "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"Saved capture data to {save_dir}, press 'c' to capture again or 'q' to quit.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect two AprilTags and compute camera pose relative to their midpoint."
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.06,
        help="Physical size of AprilTags in meters (default: 0.06).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./calibration/third_tag/",
        help="Directory to save the calibration results.",
    )
    parser.add_argument(
        "--third-tag-dir",
        type=str,
        default="./calibration/third_tag/",
        help="Directory to load the third tag pose calibration results.",
    )
    parser.add_argument(
        '--mode',
        type=int,
        default=0,
    )
    args = parser.parse_args()
    if args.mode == 0:
        calibrate_third_tag_pose(
            args.tag_size,
            args.save_dir,
        )
    elif args.mode == 1:
        calibrate_camera_pose_from_third_tag(
            args.tag_size,
            args.third_tag_dir,
            args.save_dir,
        )
