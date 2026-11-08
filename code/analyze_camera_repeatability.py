
from pathlib import Path
import json
import sys

import cv2
import numpy as np


# Project and experiment settings
project_root = Path(__file__).resolve().parent.parent

pose_label = sys.argv[1] if len(sys.argv) == 2 else "middle"

if pose_label not in ("near", "middle", "far"):
    raise ValueError("Pose label must be near, middle, or far.")

data_root = (
    project_root
    / "data/raw/camera_repeatability/run_01"
    / pose_label
)

if not data_root.is_dir():
    raise FileNotFoundError(data_root)

# Select the most recent acquisition session
sessions = sorted(
    path for path in data_root.iterdir() if path.is_dir()
)

if not sessions:
    raise RuntimeError("No repeatability sessions found.")

session_dir = sessions[-1]
print("Processing session:", session_dir)

# Check the saved observation pairs
image_files = sorted(session_dir.glob("*.png"))
json_files = sorted(session_dir.glob("*.json"))

image_ids = {path.stem for path in image_files}
json_ids = {path.stem for path in json_files}

if image_ids != json_ids:
    raise RuntimeError("PNG and JSON filenames do not match.")

if len(json_files) != 30:
    raise RuntimeError(
        f"Expected 30 observations, found {len(json_files)}."
    )

# Load camera intrinsic calibration
intrinsics_path = (
    project_root
    / "configs/lab_camera_01_intrinsics_23p5mm_preliminary.npz"
)

with np.load(intrinsics_path) as calibration:
    camera_matrix = calibration["camera_matrix"]
    dist_coeffs = calibration["dist_coeffs"]
    board_size = tuple(int(x) for x in calibration["board_size"])
    square_size_mm = float(calibration["square_size_mm"])
    image_size = tuple(int(x) for x in calibration["image_size"])

# Generate checkerboard 3D coordinates
object_points = np.zeros(
    (board_size[0] * board_size[1], 3),
    dtype=np.float32,
)

object_points[:, :2] = (
    np.mgrid[0:board_size[0], 0:board_size[1]]
    .T.reshape(-1, 2)
    * square_size_mm
)

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

positions = []
rotations = []
rms_errors = []
joint_states = []
samples = []

# Process each stationary image
for json_path in json_files:
    with json_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    image_path = session_dir / metadata["image_filename"]

    if image_path.stem != json_path.stem:
        raise RuntimeError("Observation ID mismatch.")

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Cannot read {image_path.name}")

    if (image.shape[1], image.shape[0]) != image_size:
        raise RuntimeError("Unexpected image resolution.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, board_size
    )

    if not found:
        raise RuntimeError(
            f"Checkerboard detection failed: {image_path.name}"
        )

    refined = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        refined,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        raise RuntimeError(f"Pose estimation failed: {image_path.name}")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    position_mm = tvec.reshape(3)

    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec,
        camera_matrix, dist_coeffs
    )

    corner_errors = np.linalg.norm(
        refined.reshape(-1, 2) - projected.reshape(-1, 2),
        axis=1,
    )

    rms_px = float(np.sqrt(np.mean(corner_errors**2)))

    positions.append(position_mm)
    rotations.append(rotation_matrix)
    rms_errors.append(rms_px)

    joints = np.asarray(
        metadata["robot_after"]["joint_positions_rad"],
        dtype=np.float64,
    )
    joint_states.append(joints)

    samples.append({
        "observation_id": metadata["observation_id"],
        "position_C_P_mm": position_mm.tolist(),
        "rvec_C_P_rad": rvec.reshape(3).tolist(),
        "reprojection_rms_px": rms_px,
    })

# Calculate position statistics
positions = np.asarray(positions)
rotations = np.asarray(rotations)
rms_errors = np.asarray(rms_errors)
joint_states = np.asarray(joint_states)

mean_position = np.mean(positions, axis=0)
position_std = np.std(positions, axis=0, ddof=1)

position_deviations = np.linalg.norm(
    positions - mean_position, axis=1
)

position_rms_3d = float(
    np.sqrt(np.mean(position_deviations**2))
)

# Calculate the mean orientation using rotation matrices
U, _, Vt = np.linalg.svd(
    np.mean(rotations, axis=0)
)

correction = np.diag([
    1.0, 1.0, np.linalg.det(U @ Vt)
])

mean_rotation = U @ correction @ Vt

orientation_errors_deg = []

for rotation in rotations:
    relative = mean_rotation.T @ rotation

    cosine = np.clip(
        (np.trace(relative) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    angle_deg = np.degrees(np.arccos(cosine))
    orientation_errors_deg.append(angle_deg)

orientation_errors_deg = np.asarray(orientation_errors_deg)

orientation_rms_deg = float(
    np.sqrt(np.mean(orientation_errors_deg**2))
)

# Compare recorded robot joint positions
joint_drift = np.abs(joint_states - joint_states[0])
max_joint_drift = float(np.max(joint_drift))

# Display results
print("\nSTATIONARY CAMERA REPEATABILITY")
print("Pose:", pose_label)
print("Observations:", len(positions))

print("\nMean checkerboard position (mm):")
print(np.round(mean_position, 4))

print("\nPosition standard deviation (mm):")
print("X:", float(position_std[0]))
print("Y:", float(position_std[1]))
print("Z:", float(position_std[2]))

print("\n3D position RMS deviation (mm):", position_rms_3d)
print("Maximum 3D deviation (mm):", float(np.max(position_deviations)))

print("\nOrientation RMS deviation (deg):", orientation_rms_deg)
print("Maximum orientation deviation (deg):",
      float(np.max(orientation_errors_deg)))

print("\nMean reprojection RMS (px):", float(np.mean(rms_errors)))
print("Maximum reprojection RMS (px):", float(np.max(rms_errors)))

print("\nMaximum recorded joint drift (rad):", max_joint_drift)

# Save the results
output_dir = (
    project_root
    / "data/processed/camera_repeatability/run_01"
    / pose_label
    / session_dir.name
)
output_dir.mkdir(parents=True, exist_ok=True)

report = {
    "pose_label": pose_label,
    "session_id": session_dir.name,
    "image_count": len(positions),
    "mean_position_C_P_mm": mean_position.tolist(),
    "position_std_mm": position_std.tolist(),
    "position_rms_3d_mm": position_rms_3d,
    "maximum_position_deviation_mm": float(np.max(position_deviations)),
    "orientation_rms_deg": orientation_rms_deg,
    "maximum_orientation_deviation_deg": float(np.max(orientation_errors_deg)),
    "mean_reprojection_rms_px": float(np.mean(rms_errors)),
    "maximum_reprojection_rms_px": float(np.max(rms_errors)),
    "maximum_joint_drift_rad": max_joint_drift,
    "samples": samples,
}

output_path = output_dir / "repeatability_results.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("\nSaved analysis:", output_path)
