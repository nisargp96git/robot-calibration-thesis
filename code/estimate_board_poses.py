
from pathlib import Path
import json

import cv2
import numpy as np


# Project directories
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data/raw/eye_to_hand/run_01"
output_dir = project_root / "data/processed/eye_to_hand/run_01"
output_dir.mkdir(parents=True, exist_ok=True)

intrinsics_path = (
    project_root
    / "configs/lab_camera_01_intrinsics_23p5mm_preliminary.npz"
)

# Load previously calibrated camera parameters
with np.load(intrinsics_path) as calibration:
    camera_matrix = calibration["camera_matrix"]
    dist_coeffs = calibration["dist_coeffs"]
    board_size = tuple(int(x) for x in calibration["board_size"])
    square_size_mm = float(calibration["square_size_mm"])
    image_size = tuple(int(x) for x in calibration["image_size"])

# Generate 3D checkerboard corner coordinates in millimetres
object_points = np.zeros(
    (board_size[0] * board_size[1], 3),
    dtype=np.float32,
)

object_points[:, :2] = (
    np.mgrid[0:board_size[0], 0:board_size[1]]
    .T.reshape(-1, 2)
    * square_size_mm
)

# Subpixel corner refinement settings
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

results = []

for json_path in sorted(data_dir.glob("*.json")):
    with json_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    image_path = data_dir / metadata["image_filename"]
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]
    if (width, height) != image_size:
        raise RuntimeError(f"Unexpected image size: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, board_size
    )

    if not found:
        raise RuntimeError(f"Corners not detected: {image_path}")

    refined_corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )

    # Estimate checkerboard pose in camera coordinates
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        refined_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        raise RuntimeError(f"solvePnP failed: {image_path}")

    # Project the estimated 3D corners back onto the image
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )

    observed_xy = refined_corners.reshape(-1, 2)
    projected_xy = projected.reshape(-1, 2)

    corner_errors = np.linalg.norm(
        observed_xy - projected_xy, axis=1
    )

    rms_px = float(np.sqrt(np.mean(corner_errors**2)))
    max_error_px = float(np.max(corner_errors))

    result = {
        "observation_id": metadata["observation_id"],
        "metadata_filename": json_path.name,
        "image_filename": image_path.name,
        "rvec_C_P_rad": rvec.reshape(-1).tolist(),
        "tvec_C_P_mm": tvec.reshape(-1).tolist(),
        "reprojection_rms_px": rms_px,
        "reprojection_max_px": max_error_px,
    }
    results.append(result)

    print(
        f"{image_path.name}: "
        f"RMS = {rms_px:.4f} px, "
        f"maximum = {max_error_px:.4f} px"
    )

# Save results without changing the original observations
output_path = output_dir / "board_poses.json"

report = {
    "intrinsics_file": str(intrinsics_path.relative_to(project_root)),
    "board_size": list(board_size),
    "square_size_mm": square_size_mm,
    "observations": results,
}

with output_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

errors = np.array([
    item["reprojection_rms_px"] for item in results
])

print("\nSUMMARY")
print("Processed observations:", len(results))
print("Mean RMS (px):", float(np.mean(errors)))
print("Median RMS (px):", float(np.median(errors)))
print("Highest RMS (px):", float(np.max(errors)))

print("\nFive highest-error images:")
for item in sorted(
    results,
    key=lambda x: x["reprojection_rms_px"],
    reverse=True,
)[:5]:
    print(item["image_filename"], item["reprojection_rms_px"])

print("\nSaved results:", output_path)
