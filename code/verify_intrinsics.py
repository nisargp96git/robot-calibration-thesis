
# Import NumPy to load and inspect numerical calibration data.
import numpy as np

# Import Path to construct filesystem paths.
from pathlib import Path

# Determine the root directory of the thesis project.
project_root = Path(__file__).resolve().parent.parent

# Define the location of the saved laboratory camera calibration.
calibration_file = (
    project_root / "configs" / "lab_camera_01_intrinsics_23p5mm_preliminary.npz"
)

# Check that the calibration file exists before opening it.
if not calibration_file.is_file():
    raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

# Load the saved NumPy archive and close it automatically afterward.
with np.load(calibration_file, allow_pickle=False) as calibration:

    # Print the names of all arrays stored in the file.
    print("Saved parameters:", calibration.files)

    # Print the estimated intrinsic camera matrix.
    print("\nCamera matrix:")
    print(calibration["camera_matrix"])

    # Print the estimated lens distortion coefficients.
    print("\nDistortion coefficients:")
    print(calibration["dist_coeffs"])

    # Print the image resolution used for calibration.
    print("\nImage size:", calibration["image_size"])

    # Print the physical checkerboard square width in millimetres.
    print("Square size (mm):", calibration["square_size_mm"])

    # Print the overall calibration reprojection RMS error.
    print("Calibration RMS (pixels):", calibration["rms_px"])

    # Print the number of accepted calibration images.
    print("Calibration images:", calibration["image_count"])

    # Print the name of the original calibration dataset.
    print("Dataset:", calibration["dataset_name"])
