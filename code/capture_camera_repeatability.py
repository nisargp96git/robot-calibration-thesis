
from pathlib import Path
from datetime import datetime
import json
import time

import cv2
from rtde_receive import RTDEReceiveInterface

# Experiment configuration
ROBOT_IP = "192.168.100.110"
CAMERA_INDEX = 2

IMAGE_SIZE = (1920, 1080)
BOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 23.5

SAMPLES = 30
INTERVAL_S = 2.0

MAX_SPEED_RAD_S = 0.01
MAX_CHANGE_PER_CAPTURE_RAD = 0.001
MAX_DRIFT_FROM_START_RAD = 0.0002

# Locate the thesis project
project_root = Path(__file__).resolve().parent.parent

# Reference to the existing intrinsic calibration
intrinsics_file = (
    "configs/lab_camera_01_intrinsics_23p5mm_preliminary.npz"
)

# Select the stationary robot position
pose_label = input("Pose label (near/middle/far): ").strip().lower()

if pose_label not in ("near", "middle", "far"):
    raise ValueError("Choose near, middle, or far.")

# Create a separate experiment directory

session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

output_dir = (
    project_root
    / "data/raw/camera_repeatability/run_01"
    / pose_label
    / session_id
)

output_dir.mkdir(parents=True, exist_ok=True)

print("Experiment directory:", output_dir)


def read_robot_state(receiver):
    return {
        "controller_timestamp_s": receiver.getTimestamp(),
        "joint_positions_rad": receiver.getActualQ(),
        "joint_velocities_rad_s": receiver.getActualQd(),
        "flange_pose_m_rad": receiver.getActualTCPPose(),
    }


def largest_joint_change(first_positions, second_positions):
    return max(
        abs(a - b)
        for a, b in zip(first_positions, second_positions)
    )


# Initialize the camera and robot connection
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
receiver = None

try:
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    # Configure MJPG capture at 1920 x 1080
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

    # Check the camera's actual resolution
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {width} x {height}")

    if (width, height) != IMAGE_SIZE:
        raise RuntimeError("Camera resolution is incorrect.")

    # Connect to the robot without commanding movement
    receiver = RTDEReceiveInterface(ROBOT_IP)
    print("Connected to the UR robot.")

    # Discard a few initial camera frames
    for _ in range(10):
        camera.read()

    print("Live preview started.")
    print("Press Q to quit.")
    print("Automatic capture will be added in the next step.")


    # Acquisition state
    capturing = False
    saved_count = 0
    next_capture_time = None
    start_joint_positions = None

    print("Press S to start 30 stationary captures.")
    print("Press Q to quit.")

    while True:
        # Read a frame for the live preview
        ok, frame = camera.read()

        if not ok:
            raise RuntimeError("Could not read a camera frame.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, BOARD_SIZE
        )

        preview = frame.copy()

        if found:
            cv2.drawChessboardCorners(
                preview, BOARD_SIZE, corners, found
            )

        if capturing:
            status = f"Recording: {saved_count}/{SAMPLES}"
        else:
            status = "Press S to start"

        cv2.putText(
            preview,
            status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if found else (0, 0, 255),
            2,
        )

        cv2.imshow(
            "Camera Repeatability Experiment",
            preview
        )

        key = cv2.waitKey(1) & 0xFF

        # Quit when Q is pressed
        if key == ord("q"):
            print("Experiment stopped by user.")
            break

        # Start recording when S is pressed
        if key == ord("s") and not capturing:
            if not found:
                print("Checkerboard not detected.")
                continue

            start_state = read_robot_state(receiver)

            start_speed = max(
                abs(value)
                for value in start_state["joint_velocities_rad_s"]
            )

            if start_speed > MAX_SPEED_RAD_S:
                print("Robot is moving. Recording not started.")
                continue

            start_joint_positions = start_state["joint_positions_rad"]
            next_capture_time = time.monotonic()
            capturing = True

            print("Starting stationary repeatability experiment.")

        # Continue displaying the preview until recording starts
        if not capturing:
            continue

        # Wait until the next capture time
        if time.monotonic() < next_capture_time:
            continue

        next_capture_time = time.monotonic() + INTERVAL_S

        # Read the robot state immediately before acquisition
        before = read_robot_state(receiver)

        # Acquire the image to be saved
        image_ok, image = camera.read()
        image_time = datetime.now().astimezone().isoformat()

        # Read the robot state immediately after acquisition
        after = read_robot_state(receiver)

        if not image_ok:
            raise RuntimeError("Image acquisition failed.")

        if image.shape[:2] != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
            raise RuntimeError("Unexpected captured image resolution.")

        # Check sampled robot velocities
        velocities = (
            before["joint_velocities_rad_s"]
            + after["joint_velocities_rad_s"]
        )

        max_speed = max(abs(value) for value in velocities)

        # Check robot movement during image acquisition
        max_change = largest_joint_change(
            before["joint_positions_rad"],
            after["joint_positions_rad"],
        )

        # Check displacement from the starting joint configuration
        drift_before = largest_joint_change(
            start_joint_positions,
            before["joint_positions_rad"],
        )

        drift_after = largest_joint_change(
            start_joint_positions,
            after["joint_positions_rad"],
        )

        max_drift = max(drift_before, drift_after)

        if (
            max_speed > MAX_SPEED_RAD_S
            or max_change > MAX_CHANGE_PER_CAPTURE_RAD
            or max_drift > MAX_DRIFT_FROM_START_RAD
        ):
            raise RuntimeError(
                "Robot movement threshold exceeded. "
                "Recording stopped to protect the dataset."
            )

        # Verify the checkerboard in the actual saved image
        gray_saved = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )

        detected, saved_corners = cv2.findChessboardCorners(
            gray_saved, BOARD_SIZE
        )

        if not detected or len(saved_corners) != 54:
            print("Checkerboard not detected. Retrying.")
            continue

        # Generate a unique identifier
        observation_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        image_path = output_dir / f"{observation_id}.png"
        metadata_path = output_dir / f"{observation_id}.json"

        # Prepare the measurement metadata
        metadata = {
            "observation_id": observation_id,
            "pose_label": pose_label,
            "session_id": session_id,
            "image_filename": image_path.name,
            "image_time_host_approx": image_time,
            "camera_index": CAMERA_INDEX,
            "board_size": BOARD_SIZE,
            "square_size_mm": SQUARE_SIZE_MM,
            "intrinsics_file": intrinsics_file,
            "robot_before": before,
            "robot_after": after,
            "maximum_joint_speed_rad_s": max_speed,
            "maximum_joint_change_rad": max_change,
            "maximum_drift_from_start_rad": max_drift,
        }

        # Save the original, unannotated image
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError("Failed to save the image.")

        # Save the associated robot measurements
        with metadata_path.open("x", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        saved_count += 1

        print(
            f"Saved {saved_count}/{SAMPLES}: "
            f"{observation_id}"
        )

        # Finish after 30 accepted observations
        if saved_count >= SAMPLES:
            print("Repeatability acquisition completed.")
            break


finally:
    camera.release()

    if receiver is not None:
        receiver.disconnect()

    cv2.destroyAllWindows()

    print("Camera and robot connections closed.")
