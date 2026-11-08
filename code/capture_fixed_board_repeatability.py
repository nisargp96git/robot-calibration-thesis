
from pathlib import Path
from datetime import datetime

import json
import time
import cv2

# Camera configuration
CAMERA_INDEX = 2
IMAGE_SIZE = (1920, 1080)

# Checkerboard configuration
BOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 23.5

# Acquisition configuration
SAMPLES = 30
INTERVAL_S = 2.0

# Expected intrinsic calibration
INTRINSICS_FILE = (
    "configs/lab_camera_01_intrinsics_23p5mm_preliminary.npz"
)

# Locate the thesis project
project_root = Path(__file__).resolve().parent.parent

# Verify the intrinsic calibration file exists
intrinsics_path = project_root / INTRINSICS_FILE

if not intrinsics_path.is_file():
    raise FileNotFoundError(intrinsics_path)

# Select the measurement position
pose_label = input(
    "Checkerboard position (near/middle/far): "
).strip().lower()

if pose_label not in ("near", "middle", "far"):
    raise ValueError("Choose near, middle, or far.")

# Give every acquisition session a unique identifier
session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

# Construct the output directory
output_dir = (
    project_root
    / "data/raw/fixed_board_repeatability/run_01"
    / pose_label
    / session_id
)

output_dir.mkdir(parents=True, exist_ok=True)

print("Saving observations to:", output_dir)

# Initialize the webcam
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

try:
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    # Request MJPG format
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    # Request the calibrated image resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

    # Discard initial frames
    for _ in range(10):
        camera.read()

    print("Camera initialized.")
    print("Press Q in the preview window to quit.")

    # Display the live checkerboard image

    # Automatic acquisition state
    capturing = False
    saved_count = 0
    next_capture_time = None

    print("Press S to start capturing 30 images.")
    print("Press Q to quit.")

    while True:
        # Read one original camera frame
        ok, frame = camera.read()
        frame_time = datetime.now().astimezone().isoformat()

        if not ok:
            raise RuntimeError("Could not read a camera frame.")

        if frame.shape[:2] != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
            raise RuntimeError("Incorrect image resolution.")

        # Detect the checkerboard
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, BOARD_SIZE
        )

        corner_count = len(corners) if corners is not None else 0
        valid_board = found and corner_count == 54

        # Make a separate image for the live preview
        preview = frame.copy()

        if valid_board:
            cv2.drawChessboardCorners(
                preview, BOARD_SIZE, corners, found
            )

        if capturing:
            status = f"Recording: {saved_count}/{SAMPLES}"
        else:
            status = "Press S to start"

        color = (0, 255, 0) if valid_board else (0, 0, 255)

        cv2.putText(
            preview,
            status,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            2
        )

        display = cv2.resize(preview, (960, 540))
        cv2.imshow("Fixed Board Repeatability", display)

        key = cv2.waitKey(1) & 0xFF

        # Stop when Q is pressed
        if key == ord("q"):
            print("Acquisition stopped by user.")
            break

        # Start acquisition when S is pressed
        if key == ord("s") and not capturing:
            if not valid_board:
                print("Checkerboard not detected. Cannot start.")
                continue

            capturing = True
            next_capture_time = time.monotonic()

            print("Starting automatic image acquisition.")

        # Continue the preview until acquisition starts
        if not capturing:
            continue

        # Wait until the next capture is due
        if time.monotonic() < next_capture_time:
            continue

        # Reject images without all 54 corners
        if not valid_board:
            print("Checkerboard not detected. Retrying.")
            next_capture_time = time.monotonic() + INTERVAL_S
            continue

        # Create a unique observation identifier
        observation_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        image_path = output_dir / f"{observation_id}.png"
        metadata_path = output_dir / f"{observation_id}.json"

        # Record the acquisition settings
        metadata = {
            "observation_id": observation_id,
            "session_id": session_id,
            "pose_label": pose_label,
            "image_filename": image_path.name,
            "image_time_host_approx": frame_time,
            "camera_index": CAMERA_INDEX,
            "image_size": list(IMAGE_SIZE),
            "board_size": list(BOARD_SIZE),
            "square_size_mm": SQUARE_SIZE_MM,
            "intrinsics_file": INTRINSICS_FILE,
            "checkerboard_detected": True,
            "corner_count": corner_count,
            "sample_number": saved_count + 1,
            "requested_interval_s": INTERVAL_S
        }

        # Save the original image without overlays
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError("Failed to save camera image.")

        # Save the corresponding JSON metadata
        with metadata_path.open("x", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        saved_count += 1

        print(
            f"Saved {saved_count}/{SAMPLES}: "
            f"{image_path.name}"
        )

        # Stop automatically after 30 accepted images
        if saved_count >= SAMPLES:
            print("All 30 images captured successfully.")
            break

        # Wait at least two seconds before the next capture
        next_capture_time = time.monotonic() + INTERVAL_S

finally:
    camera.release()
    cv2.destroyAllWindows()

    print("Camera connection closed.")
