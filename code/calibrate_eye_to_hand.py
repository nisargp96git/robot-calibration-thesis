
from pathlib import Path
import json

import cv2
import numpy as np


# 1. Project paths
project_root = Path(__file__).resolve().parent.parent

data_dir = project_root / "data/raw/eye_to_hand/run_01"

board_pose_path = (
    project_root
    / "data/processed/eye_to_hand/run_01/board_poses.json"
)

output_dir = project_root / "data/processed/eye_to_hand/run_01"
output_dir.mkdir(parents=True, exist_ok=True)


# 2. Load previously calculated checkerboard poses
with board_pose_path.open("r", encoding="utf-8") as file:
    board_data = json.load(file)

observations = sorted(
    board_data["observations"],
    key=lambda record: record["observation_id"],
)

if len(observations) != 30:
    raise RuntimeError("Expected exactly 30 observations.")

if float(board_data["square_size_mm"]) != 23.5:
    raise RuntimeError("Unexpected checkerboard square size.")

all_ids = [record["observation_id"] for record in observations]

if len(set(all_ids)) != len(all_ids):
    raise RuntimeError("Duplicate observation IDs found.")


# 3. Select calibration and validation observations
duplicate_id = observations[1]["observation_id"]

validation_indices = (4, 9, 14, 19, 24, 29)

validation_ids = {
    observations[index]["observation_id"]
    for index in validation_indices
}

training_records = [
    record for record in observations
    if record["observation_id"] not in validation_ids
    and record["observation_id"] != duplicate_id
]

if len(training_records) != 23:
    raise RuntimeError("Incorrect calibration dataset size.")

print("Calibration observations:", len(training_records))
print("Validation observations:", len(validation_ids))
print("Excluded repeated observation:", duplicate_id)


# 4. Convert rotation and translation into a 4x4 matrix
def make_transform(rotation_vector, translation):
    rvec = np.asarray(
        rotation_vector, dtype=np.float64
    ).reshape(3, 1)

    tvec = np.asarray(
        translation, dtype=np.float64
    ).reshape(3)

    rotation_matrix, _ = cv2.Rodrigues(rvec)

    transform = np.eye(4, dtype=np.float64)

    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = tvec

    return transform


# 5. Prepare the input arrays for OpenCV
R_flange_from_base = []
t_flange_from_base = []

R_camera_from_board = []
t_camera_from_board = []

training_pairs = []

for record in training_records:
    metadata_path = data_dir / record["metadata_filename"]

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata["observation_id"] != record["observation_id"]:
        raise RuntimeError("Robot-camera observation mismatch.")

    # The UR reports flange translation in metres.
    robot_pose = np.asarray(
        metadata["robot_after"]["flange_pose_m_rad"],
        dtype=np.float64,
    )

    if robot_pose.shape != (6,) or not np.all(np.isfinite(robot_pose)):
        raise RuntimeError("Invalid robot flange pose.")

    # Robot flange to base: ^B T_F, translation in mm
    T_B_F = make_transform(
        robot_pose[3:],
        robot_pose[:3] * 1000.0,
    )

    # Checkerboard to camera: ^C T_P, translation in mm
    T_C_P = make_transform(
        record["rvec_C_P_rad"],
        record["tvec_C_P_mm"],
    )

    # Invert the robot pose for eye-to-hand calibration.
    T_F_B = np.linalg.inv(T_B_F)

    R_flange_from_base.append(T_F_B[:3, :3])
    t_flange_from_base.append(
        T_F_B[:3, 3].reshape(3, 1)
    )

    R_camera_from_board.append(T_C_P[:3, :3])
    t_camera_from_board.append(
        T_C_P[:3, 3].reshape(3, 1)
    )

    training_pairs.append((T_B_F, T_C_P))


# 6. Estimate the fixed camera-to-base transformation
R_B_C, t_B_C = cv2.calibrateHandEye(
    R_flange_from_base,
    t_flange_from_base,
    R_camera_from_board,
    t_camera_from_board,
    method=cv2.CALIB_HAND_EYE_PARK,
)

T_B_C = np.eye(4, dtype=np.float64)

T_B_C[:3, :3] = R_B_C
T_B_C[:3, 3] = np.asarray(t_B_C).reshape(3)

if not np.all(np.isfinite(T_B_C)):
    raise RuntimeError("Non-finite eye-to-hand calibration result.")


# 7. Calculate the fixed flange-to-checkerboard transformation
board_mount_estimates = []

for T_B_F, T_C_P in training_pairs:
    T_F_P_i = np.linalg.inv(T_B_F) @ T_B_C @ T_C_P
    board_mount_estimates.append(T_F_P_i)

translations = np.array([
    transform[:3, 3]
    for transform in board_mount_estimates
])

mean_translation = np.mean(translations, axis=0)

mean_rotation_matrix = np.mean(
    [transform[:3, :3] for transform in board_mount_estimates],
    axis=0,
)

U, _, Vt = np.linalg.svd(mean_rotation_matrix)

correction = np.diag([
    1.0,
    1.0,
    np.linalg.det(U @ Vt),
])

mean_rotation = U @ correction @ Vt

T_F_P = np.eye(4, dtype=np.float64)
T_F_P[:3, :3] = mean_rotation
T_F_P[:3, 3] = mean_translation


# 8. Calculate preliminary calibration consistency
translation_errors = []
rotation_errors = []

for transform in board_mount_estimates:
    translation_error = np.linalg.norm(
        transform[:3, 3] - mean_translation
    )

    relative_rotation = (
        mean_rotation.T @ transform[:3, :3]
    )

    cosine = np.clip(
        (np.trace(relative_rotation) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    rotation_error = np.degrees(np.arccos(cosine))

    translation_errors.append(translation_error)
    rotation_errors.append(rotation_error)

translation_rms = float(
    np.sqrt(np.mean(np.square(translation_errors)))
)

rotation_rms = float(
    np.sqrt(np.mean(np.square(rotation_errors)))
)


# 9. Print the estimated transformations
np.set_printoptions(precision=6, suppress=True)

print("\nCAMERA TO ROBOT BASE: ^B T_C")
print(T_B_C)

print("\nCHECKERBOARD TO FLANGE: ^F T_P")
print(T_F_P)

print("\nTRAINING CONSISTENCY")
print("Translation RMS (mm):", translation_rms)
print("Translation maximum (mm):", max(translation_errors))
print("Rotation RMS (deg):", rotation_rms)
print("Rotation maximum (deg):", max(rotation_errors))


# 10. Save results for independent validation
output_path = output_dir / "eye_to_hand_initial.npz"

np.savez_compressed(
    output_path,
    T_base_camera=T_B_C,
    T_flange_board=T_F_P,
    training_ids=np.array([
        record["observation_id"]
        for record in training_records
    ]),
    validation_ids=np.array(sorted(validation_ids)),
    duplicate_id=np.array(duplicate_id),
    translation_training_rms_mm=translation_rms,
    rotation_training_rms_deg=rotation_rms,
)

print("\nSaved calibration:", output_path)
print("Validation observations were not used for fitting.")
