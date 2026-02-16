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


# Conversion between the detector's camera frame (x-right, y-front, z-down)
# and the desired robotics frame (x-front, y-right, z-up).
AXIS_CONVERSION_MATRIX = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def convert_pose_axes(rotation, translation):
    """Convert R, t from detector frame to x-front, y-right, z-up frame."""
    converted_rotation = AXIS_CONVERSION_MATRIX @ rotation
    converted_translation = AXIS_CONVERSION_MATRIX @ translation.reshape(3, 1)
    return converted_rotation, converted_translation


def get_camera_intrinsics(pipeline_profile):
    """Fetch RealSense intrinsics and build camera matrix + distortion."""
    color_stream = pipeline_profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    camera_matrix = np.array(
        [
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1],
        ]
    )
    dist_coeffs = np.array(
        [
            color_intrinsics.coeffs[0],
            color_intrinsics.coeffs[1],
            color_intrinsics.coeffs[2],
            color_intrinsics.coeffs[3],
            color_intrinsics.coeffs[4],
        ]
    )
    return camera_matrix, dist_coeffs


def detect_apriltags(image, camera_matrix, tag_size):
    """Run AprilTag detection with pose estimation."""
    if not APRILTAG_AVAILABLE:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    detector = apriltag.Detector(
        families="tagCustom48h12",
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    results = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[fx, fy, cx, cy],
        tag_size=tag_size,
    )

    detected_tags = []
    for result in results:
        if not hasattr(result, "pose_R") or not hasattr(result, "pose_t"):
            continue
        if result.pose_R is None or result.pose_t is None:
            continue

        try:
            rotation_matrix = np.array(result.pose_R, dtype=np.float64)
            rvec, _ = cv2.Rodrigues(rotation_matrix)
            tvec = np.array(result.pose_t, dtype=np.float64).reshape(3, 1)
            detected_tags.append(
                {
                    "tag_id": int(result.tag_id),
                    "rotation_matrix": rotation_matrix,
                    "rotation_vector": rvec,
                    "translation_vector": tvec,
                    "distance": float(np.linalg.norm(tvec)),
                    "corners": np.array(result.corners, dtype=np.float32).tolist(),
                    "center": np.array(result.center, dtype=np.float32).tolist(),
                }
            )
        except cv2.error:
            continue

    return detected_tags


def select_two_tags(detected_tags, expected_ids):
    """Return two tags either by ID list or nearest distance."""
    if expected_ids:
        selected = []
        for tag_id in expected_ids:
            tag = next((t for t in detected_tags if t["tag_id"] == tag_id), None)
            if tag is None:
                return None, f"Missing expected tag ID {tag_id}"
            selected.append(tag)
        return selected, None

    if len(detected_tags) < 2:
        return None, "Less than two tags detected"

    sorted_tags = sorted(detected_tags, key=lambda t: t["distance"])
    return sorted_tags[:2], None


def compute_midpoint_pose(tags):
    """Compute pose of midpoint frame in camera coordinates."""
    translations = np.stack(
        [tag["translation_vector"].reshape(3, 1) for tag in tags], axis=0
    )
    rotation_vectors = np.stack(
        [tag["rotation_vector"].reshape(3, 1) for tag in tags], axis=0
    )

    midpoint_translation = translations.mean(axis=0)
    midpoint_rvec = rotation_vectors.mean(axis=0)
    midpoint_rotation, _ = cv2.Rodrigues(midpoint_rvec)

    return midpoint_rotation, midpoint_translation


def convert_pose_from_midpoint_to_world(rotation, translation):
    translation[0] -= 0.5
    translation[2] += 0.6
    return rotation, translation


def invert_pose(rotation, translation):
    """Return inverse of pose represented by R, t."""
    rotation_inv = rotation.T
    translation_inv = -rotation_inv @ translation
    return rotation_inv, translation_inv


def log_pose(prefix, rotation, translation):
    """Pretty print pose data (x-front, y-right, z-up)."""
    print(prefix)
    print("  Rotation matrix:")
    print(rotation)
    print("  Translation (x, y, z) m:")
    print(translation.reshape(-1))


def draw_tag_axes(image, tag, camera_matrix, dist_coeffs, axis_length=0.05):
    """Overlay tag outline and axes."""
    if "corners" not in tag or "center" not in tag:
        return

    corners = np.array(tag["corners"], dtype=np.int32)
    center = np.array(tag["center"], dtype=np.int32)
    cv2.polylines(image, [corners], True, (0, 255, 0), 2)
    cv2.circle(image, tuple(center), 4, (0, 255, 0), -1)
    cv2.putText(
        image,
        f"ID:{tag['tag_id']}",
        (center[0] - 20, center[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    rvec = tag["rotation_vector"].reshape(3, 1)
    tvec = tag["translation_vector"].reshape(3, 1)
    axis_points = np.array(
        [
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
            [0, 0, -axis_length],
        ],
        dtype=np.float32,
    )

    projected, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2).astype(int)
    origin = tuple(projected[0])
    # Map detector axes (x-right, y-front, z-down) to the robotics frame (x-front, y-right, z-up).
    cv2.arrowedLine(image, origin, tuple(projected[2]), (0, 0, 255), 2)  # new x (front)
    cv2.arrowedLine(image, origin, tuple(projected[1]), (0, 255, 0), 2)  # new y (right)
    cv2.arrowedLine(image, origin, tuple(projected[4]), (255, 0, 0), 2)  # new z (up)


def draw_midpoint_axes(image, rotation, translation, camera_matrix, dist_coeffs, axis_length=0.05):
    """Draw axes for the midpoint pose."""
    axis_points = np.array(
        [
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
            [0, 0, -axis_length],
        ],
        dtype=np.float32,
    )
    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(axis_points, rvec, translation, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2).astype(int)
    origin = tuple(projected[0])
    cv2.putText(
        image,
        "Midpoint",
        (origin[0] + 5, origin[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )
    cv2.arrowedLine(image, origin, tuple(projected[2]), (0, 0, 255), 2)
    cv2.arrowedLine(image, origin, tuple(projected[1]), (0, 255, 0), 2)
    cv2.arrowedLine(image, origin, tuple(projected[4]), (255, 0, 0), 2)


def calibrate(
    tag_size,
    tag_ids,
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

            tags_of_interest, reason = select_two_tags(detections, tag_ids)
            if not tags_of_interest:
                print(f"No valid tag pair detected: {reason}")
                continue

            midpoint_rotation, midpoint_translation = compute_midpoint_pose(tags_of_interest)
            camera_rotation, camera_translation = invert_pose(
                midpoint_rotation, midpoint_translation 
            )

            print(f"Detected tags {[tag['tag_id'] for tag in tags_of_interest]}")
            for tag in tags_of_interest:
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
            )
            camera_rotation_world, camera_translation_world = convert_pose_from_midpoint_to_world(
                camera_rotation_converted, camera_translation_converted
            )

            log_pose(f"{'='*30}\nMidpoint pose in camera frame:", midpoint_rotation_converted, midpoint_translation_converted)
            log_pose(f"{'='*30}\nCamera pose in midpoint frame:", camera_rotation_converted, camera_translation_converted)
            log_pose(f"{'='*30}\nCamera pose in world frame:", camera_rotation_world, camera_translation_world)

            annotated = color_image.copy()
            for tag in tags_of_interest:
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
                    "camera_rotation_matrix_world": camera_rotation_world.tolist(),
                    "camera_translation_world": camera_translation_world.reshape(-1).tolist(),
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
        "--tag-ids",
        type=int,
        nargs=2,
        default=[0, 1],
        help="Optional IDs of the two tags to use.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save the calibration results.",
    )
    args = parser.parse_args()
    calibrate(
        args.tag_size,
        args.tag_ids,
        args.save_dir,
    )
