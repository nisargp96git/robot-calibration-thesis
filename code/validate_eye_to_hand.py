
from pathlib import Path
import json

import cv2
import numpy as np


# Locate the project files
project_root = Path(__file__).resolve().parent.parent

data_dir = project_root / "data/raw/eye_to_hand/run_01"

processed_dir = (
    project_root / "data/processed/eye_to_hand/run_01"
)

calibration_path = processed_dir / "eye_to_hand_initial.npz"
board_poses_path = processed_dir / "board_poses.json"


# Load the fixed transformations and dataset split
with np.load(calibration_path, allow_pickle=False) as calibration:
    T_B_C = calibration["T_base_camera"]
    T_F_P = calibration["T_flange_board"]

    training_ids = set(calibration["training_ids"].tolist())
    validation_ids = calibration["validation_ids"].tolist()

if training_ids.intersection(validation_ids):
    raise RuntimeError("Training and validation sets overlap.")

print("Held-out validation observations:", len(validation_ids))


# Load the checkerboard pose measurements
with board_poses_path.open("r", encoding="utf-8") as file:
    board_data = json.load(file)

board_records = {
    record["observation_id"]: record
    for record in board_data["observations"]
}


# Construct a homogeneous transformation
def make_transform(rvec, translation):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    translation = np.asarray(
        translation, dtype=np.float64
    ).reshape(3)

    rotation, _ = cv2.Rodrigues(rvec)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


# Compare the two predicted checkerboard poses
results = []

for observation_id in validation_ids:
    record = board_records[observation_id]

    metadata_path = data_dir / record["metadata_filename"]

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata["observation_id"] != observation_id:
        raise RuntimeError("Observation ID mismatch.")

    robot_pose = np.asarray(
        metadata["robot_after"]["flange_pose_m_rad"],
        dtype=np.float64,
    )

    # The UR reports translation in metres.
    T_B_F = make_transform(
        robot_pose[3:],
        robot_pose[:3] * 1000.0,
    )

    # solvePnP translation is already in millimetres.
    T_C_P = make_transform(
        record["rvec_C_P_rad"],
        record["tvec_C_P_mm"],
    )

    # Predict the board pose through both paths.
    T_B_P_robot = T_B_F @ T_F_P
    T_B_P_camera = T_B_C @ T_C_P

    # Translation residual expressed in robot-base axes.
    translation_difference = (
        T_B_P_camera[:3, 3] - T_B_P_robot[:3, 3]
    )

    translation_error = float(
        np.linalg.norm(translation_difference)
    )

    # Angular difference between the two rotations.
    relative_rotation = (
        T_B_P_robot[:3, :3].T
        @ T_B_P_camera[:3, :3]
    )

    cosine = np.clip(
        (np.trace(relative_rotation) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    rotation_error = float(
        np.degrees(np.arccos(cosine))
    )

    results.append({
        "observation_id": observation_id,
        "translation_error_mm": translation_error,
        "translation_components_mm":
            translation_difference.tolist(),
        "rotation_error_deg": rotation_error,
        "reprojection_rms_px":
            record["reprojection_rms_px"],
    })

    print(
        f"{observation_id}: "
        f"translation = {translation_error:.4f} mm, "
        f"rotation = {rotation_error:.4f} deg"
    )


# Summarize held-out validation errors
translation_errors = np.array([
    result["translation_error_mm"] for result in results
])

rotation_errors = np.array([
    result["rotation_error_deg"] for result in results
])

translation_rms = float(
    np.sqrt(np.mean(translation_errors**2))
)

rotation_rms = float(
    np.sqrt(np.mean(rotation_errors**2))
)

print("\nVALIDATION SUMMARY")
print("Observations:", len(results))
print("Translation RMS (mm):", translation_rms)
print("Translation median (mm):", float(np.median(translation_errors)))
print("Translation maximum (mm):", float(np.max(translation_errors)))

print("Rotation RMS (deg):", rotation_rms)
print("Rotation median (deg):", float(np.median(rotation_errors)))
print("Rotation maximum (deg):", float(np.max(rotation_errors)))


# Save validation results
report = {
    "validation_observations": len(results),
    "translation_rms_mm": translation_rms,
    "rotation_rms_deg": rotation_rms,
    "results": results,
}

output_path = processed_dir / "eye_to_hand_validation.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("\nSaved validation:", output_path)
