
from pathlib import Path
import cv2
import numpy as np

# Project and dataset paths
project_root = Path(__file__).resolve().parent.parent

calibration_file = (
    project_root / "configs" / "lab_camera_01_intrinsics_23p5mm_preliminary.npz"
)

validation_dir = (
    project_root / "data/raw/camera_01/intrinsic/validation_run_01"
)

# Load the saved intrinsic parameters.
with np.load(calibration_file, allow_pickle=False) as calibration:
    camera_matrix = calibration["camera_matrix"].copy()
    dist_coeffs = calibration["dist_coeffs"].copy()
    board_size = tuple(int(v) for v in calibration["board_size"])
    square_size = float(calibration["square_size_mm"])
    image_size = tuple(int(v) for v in calibration["image_size"])
    calibration_rms = float(calibration["rms_px"])

# Generate 3D checkerboard reference points in millimetres.
columns, rows = board_size

object_points = np.zeros((columns * rows, 3), dtype=np.float32)

object_points[:, :2] = (
    np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
)

# Subpixel corner-refinement termination criteria.
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# Find validation images.
image_paths = sorted(validation_dir.glob("image_*.png"))

if not image_paths:
    raise FileNotFoundError("No validation images were found.")

total_squared_error = 0.0
total_points = 0
accepted_images = 0

print("Validation images found:", len(image_paths))
print("Calibration RMS:", calibration_rms, "pixels")
print("Square size:", square_size, "mm")

# Process every validation image.
for image_path in image_paths:

    image = cv2.imread(str(image_path))

    if image is None:
        print(image_path.name, ": Could not read image")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if (gray.shape[1], gray.shape[0]) != image_size:
        print(image_path.name, ": Image resolution mismatch")
        continue

    found, corners = cv2.findChessboardCorners(
        gray, board_size, None
    )

    if not found:
        print(image_path.name, ": Checkerboard not detected")
        continue

    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )

    # Estimate board pose without changing camera intrinsics.
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print(image_path.name, ": Pose estimation failed")
        continue

    # Reproject known 3D corners using the estimated pose.
    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    # Compute squared 2D reprojection distances.
    residuals = (
        corners.reshape(-1, 2)
        - projected_points.reshape(-1, 2)
    )

    squared_errors = np.sum(residuals ** 2, axis=1)

    image_rms = np.sqrt(np.mean(squared_errors))

    total_squared_error += np.sum(squared_errors)
    total_points += len(squared_errors)
    accepted_images += 1

    print(f"{image_path.name}: {image_rms:.4f} pixels")

# Calculate the overall RMS across all accepted corners.
if total_points == 0:
    raise RuntimeError("No validation images could be processed.")

validation_rms = np.sqrt(total_squared_error / total_points)

print("\nValidation images accepted:", accepted_images)
print("Total corner observations:", total_points)
print("Calibration RMS:", calibration_rms, "pixels")
print("Validation RMS:", validation_rms, "pixels")
