
from pathlib import Path
import json

import cv2
import numpy as np


# --------------------------------------------------
# 1. Project paths and experiment configuration
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

raw_dir = (
    project_root
    / "data/raw/fixed_board_repeatability/run_01"
)

output_dir = (
    project_root
    / "data/processed/fixed_board_repeatability/run_01"
)

intrinsics_path = (
    project_root
    / "configs/lab_camera_01_intrinsics_23p5mm_preliminary.npz"
)

board_size = (9, 6)
expected_samples = 30
positions = ("near", "middle", "far")


# --------------------------------------------------
# 2. Load the existing camera calibration
# --------------------------------------------------

with np.load(intrinsics_path, allow_pickle=False) as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    image_size = tuple(int(v) for v in data["image_size"])
    calibrated_board_size = tuple(
        int(v) for v in data["board_size"]
    )
    square_size_mm = float(data["square_size_mm"])

if calibrated_board_size != board_size:
    raise ValueError("Checkerboard pattern does not match calibration.")

if not np.isclose(square_size_mm, 23.5):
    raise ValueError("Unexpected checkerboard square size.")

print("Camera calibration loaded.")
print("Image size:", image_size)
print("Checkerboard:", board_size)
print("Square size (mm):", square_size_mm)


# --------------------------------------------------
# 3. Define the checkerboard's 3D corner coordinates
# --------------------------------------------------

object_points = np.zeros(
    (board_size[0] * board_size[1], 3),
    dtype=np.float32
)

object_points[:, :2] = (
    np.mgrid[0:board_size[0], 0:board_size[1]]
    .T.reshape(-1, 2)
    * square_size_mm
)

corner_criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# --------------------------------------------------
# 4. Helper functions for rotation calculations
# --------------------------------------------------

def rotation_difference_deg(rotation_a, rotation_b):
    relative_rotation = rotation_a.T @ rotation_b

    cosine = np.clip(
        (np.trace(relative_rotation) - 1.0) / 2.0,
        -1.0,
        1.0
    )

    return float(np.degrees(np.arccos(cosine)))


def average_rotation(rotation_matrices):
    matrix_sum = np.sum(rotation_matrices, axis=0)

    u, _, vt = np.linalg.svd(matrix_sum)

    correction = np.diag([
        1.0,
        1.0,
        np.linalg.det(u @ vt)
    ])

    return u @ correction @ vt


# --------------------------------------------------
# 5. Process one stationary checkerboard position
# --------------------------------------------------

def analyze_position(position):

    position_dir = raw_dir / position

    # Ignore empty directories from earlier preview tests.
    sessions = [
        directory
        for directory in sorted(position_dir.iterdir())
        if directory.is_dir()
        and any(directory.glob("*.png"))
    ]

    if len(sessions) != 1:
        raise RuntimeError(
            f"{position}: expected one nonempty session, "
            f"found {len(sessions)}."
        )

    session_dir = sessions[0]
    image_paths = sorted(session_dir.glob("*.png"))
    metadata_paths = sorted(session_dir.glob("*.json"))

    image_ids = {path.stem for path in image_paths}
    metadata_ids = {path.stem for path in metadata_paths}

    if image_ids != metadata_ids:
        raise RuntimeError(f"{position}: image/JSON pairing mismatch.")

    if len(image_paths) != expected_samples:
        raise RuntimeError(
            f"{position}: expected {expected_samples} images, "
            f"found {len(image_paths)}."
        )

    translations = []
    rotations = []
    reprojection_errors = []
    observations = []

    print(f"\nProcessing {position}: {len(image_paths)} images")

    for image_path in image_paths:

        # Read matching metadata.
        metadata_path = image_path.with_suffix(".json")

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        if metadata["image_filename"] != image_path.name:
            raise RuntimeError("Image filename mismatch.")

        if tuple(metadata["board_size"]) != board_size:
            raise RuntimeError("Checkerboard pattern mismatch.")

        if not np.isclose(
            float(metadata["square_size_mm"]),
            square_size_mm
        ):
            raise RuntimeError("Checkerboard pitch mismatch.")

        # Read the original image.
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Cannot read {image_path.name}")

        if image.shape[1::-1] != image_size:
            raise RuntimeError("Image resolution mismatch.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Locate all 54 checkerboard corners.
        found, corners = cv2.findChessboardCorners(
            gray, board_size
        )

        if not found or len(corners) != 54:
            raise RuntimeError(
                f"Checkerboard not detected: {image_path.name}"
            )

        # Refine corner coordinates to subpixel precision.
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            corner_criteria
        )

        # Estimate checkerboard pose in camera coordinates.
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            raise RuntimeError(
                f"solvePnP failed: {image_path.name}"
            )

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_mm = tvec.reshape(3)

        # Reproject the estimated 3D corners into the image.
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )

        pixel_errors = (
            projected.reshape(-1, 2)
            - corners.reshape(-1, 2)
        )

        reprojection_rms = float(
            np.sqrt(
                np.mean(np.sum(pixel_errors**2, axis=1))
            )
        )

        translations.append(translation_mm)
        rotations.append(rotation_matrix)
        reprojection_errors.append(reprojection_rms)

        observations.append({
            "image_filename": image_path.name,
            "translation_C_P_mm": translation_mm.tolist(),
            "rvec_C_P_rad": rvec.reshape(3).tolist(),
            "reprojection_rms_px": reprojection_rms
        })

    # Convert lists into NumPy arrays.
    translations = np.asarray(translations)
    rotations = np.asarray(rotations)
    reprojection_errors = np.asarray(reprojection_errors)

    # Calculate translation repeatability.
    mean_translation = np.mean(translations, axis=0)

    translation_std = np.std(
        translations, axis=0, ddof=1
    )

    deviations = translations - mean_translation

    translation_rms = float(
        np.sqrt(np.mean(np.sum(deviations**2, axis=1)))
    )

    # Calculate rotational repeatability.
    mean_rotation = average_rotation(rotations)

    rotation_errors = np.array([
        rotation_difference_deg(mean_rotation, rotation)
        for rotation in rotations
    ])

    rotation_rms = float(
        np.sqrt(np.mean(rotation_errors**2))
    )

    summary = {
        "position": position,
        "session_directory": str(session_dir),
        "image_count": len(image_paths),
        "mean_translation_C_P_mm": mean_translation.tolist(),
        "translation_std_xyz_mm": translation_std.tolist(),
        "translation_3d_rms_mm": translation_rms,
        "rotation_rms_deg": rotation_rms,
        "reprojection_rms_mean_px": float(
            np.mean(reprojection_errors)
        ),
        "reprojection_rms_max_px": float(
            np.max(reprojection_errors)
        ),
        "observations": observations
    }

    print("Mean position XYZ (mm):", mean_translation)
    print("Position standard deviation XYZ (mm):", translation_std)
    print("3D position RMS (mm):", translation_rms)
    print("Rotation RMS (deg):", rotation_rms)
    print("Mean reprojection RMS (px):",
          summary["reprojection_rms_mean_px"])

    return summary


# --------------------------------------------------
# 6. Analyze all three positions and save results
# --------------------------------------------------

results = {
    "experiment": "fixed_board_repeatability",
    "intrinsics_file": str(intrinsics_path),
    "positions": {}
}

for position in positions:
    results["positions"][position] = analyze_position(position)

output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "repeatability_results.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("\nSUMMARY")
print("Position | 3D RMS (mm) | Rotation RMS (deg)")

for position in positions:
    result = results["positions"][position]

    print(
        f"{position:6s} | "
        f"{result['translation_3d_rms_mm']:.6f} | "
        f"{result['rotation_rms_deg']:.6f}"
    )

print("\nSaved results:", output_path)
