
from pathlib import Path
import json
import cv2

# Locate the thesis project and observation directory.
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data/raw/eye_to_hand/run_01"

# Expected calibration configuration.
board_size = (9, 6)
square_size_mm = 23.5
image_height = 1080
image_width = 1920

intrinsics_file = (
    "configs/lab_camera_01_intrinsics_23p5mm_preliminary.npz"
)

if not (project_root / intrinsics_file).is_file():
    raise FileNotFoundError(intrinsics_file)

# Find the recorded images and robot measurements.
image_files = sorted(data_dir.glob("*.png"))
json_files = sorted(data_dir.glob("*.json"))

image_ids = {path.stem for path in image_files}
json_ids = {path.stem for path in json_files}

print("PNG files:", len(image_files))
print("JSON files:", len(json_files))
print("Images without JSON:", sorted(image_ids - json_ids))
print("JSON without images:", sorted(json_ids - image_ids))
print()

passed = 0
failed = 0

for json_path in json_files:
    problems = []

    try:
        with json_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        # Check the image and its metadata.
        expected_image = json_path.stem + ".png"
        image_path = data_dir / expected_image

        if metadata.get("image_filename") != expected_image:
            problems.append("Image filename mismatch")

        if metadata.get("board_size") != list(board_size):
            problems.append("Incorrect board dimensions")

        if metadata.get("square_size_mm") != square_size_mm:
            problems.append("Incorrect square size")

        if metadata.get("intrinsics_file") != intrinsics_file:
            problems.append("Incorrect intrinsics reference")

        # Decode and examine the saved photograph.
        image = cv2.imread(str(image_path))

        if image is None:
            problems.append("Image missing or unreadable")
        else:
            if image.shape[:2] != (image_height, image_width):
                problems.append("Incorrect image resolution")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(
                gray, board_size
            )

            if not found or corners is None or len(corners) != 54:
                problems.append("54 checkerboard corners not detected")

        # Inspect robot measurements before and after capture.
        before = metadata["robot_before"]
        after = metadata["robot_after"]

        for key in (
            "joint_positions_rad",
            "joint_velocities_rad_s",
            "flange_pose_m_rad",
        ):
            if len(before[key]) != 6 or len(after[key]) != 6:
                problems.append("Invalid robot array: " + key)

        speeds = (
            before["joint_velocities_rad_s"]
            + after["joint_velocities_rad_s"]
        )
        max_speed = max(abs(value) for value in speeds)

        joint_changes = [
            abs(a - b)
            for a, b in zip(
                before["joint_positions_rad"],
                after["joint_positions_rad"],
            )
        ]
        max_change = max(joint_changes)

        if max_speed > 0.01:
            problems.append("Robot velocity exceeds threshold")

        if max_change > 0.001:
            problems.append("Joint change exceeds threshold")

        if (
            after["controller_timestamp_s"]
            <= before["controller_timestamp_s"]
        ):
            problems.append("Invalid controller timestamp order")

    except (OSError, ValueError, KeyError, TypeError) as error:
        problems.append(str(error))

    if problems:
        failed += 1
        print("CHECK:", json_path.stem)
        for problem in problems:
            print("  -", problem)
    else:
        passed += 1
        print("PASS:", json_path.stem)

print()
print("AUDIT SUMMARY")
print("Passed:", passed)
print("Flagged:", failed)
print("Total JSON observations:", len(json_files))
