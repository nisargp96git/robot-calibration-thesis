
from pathlib import Path
from datetime import datetime
import json
import cv2
from rtde_receive import RTDEReceiveInterface

# Laboratory equipment and checkerboard configuration.
robot_ip = "192.168.100.110"
camera_index = 2
board_size = (9, 6)
square_size_mm = 23.5

# Create a separate eye-to-hand calibration dataset.
project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "data/raw/eye_to_hand/run_01"
output_dir.mkdir(parents=True, exist_ok=True)

# Read the current robot state without commanding movement.
def read_robot_state(receiver):
    return {
        "controller_timestamp_s": receiver.getTimestamp(),
        "joint_positions_rad": receiver.getActualQ(),
        "joint_velocities_rad_s": receiver.getActualQd(),
        "flange_pose_m_rad": receiver.getActualTCPPose()
    }

camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
receiver = None

try:
    if not camera.isOpened():
        raise RuntimeError("Cannot open laboratory webcam")

    # Use the same image format as intrinsic calibration.
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    receiver = RTDEReceiveInterface(robot_ip)

    # Discard the first camera frames for exposure stabilization.
    for _ in range(10):
        camera.read()

    print("Press S to save one observation. Press Q to quit.")

    while True:
        ok, frame = camera.read()

        if not ok:
            raise RuntimeError("Camera frame acquisition failed")

        if frame.shape[:2] != (1080, 1920):
            raise RuntimeError("Unexpected camera resolution")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, board_size, None
        )

        preview = frame.copy()

        if found:
            cv2.drawChessboardCorners(
                preview, board_size, corners, found
            )

        cv2.imshow("Eye-to-hand acquisition", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            if not found:
                print("Checkerboard not detected. Image not saved.")
                continue

            # Bracket the exposure with robot-state measurements.
            before = read_robot_state(receiver)
            ok, image = camera.read()
            image_time = datetime.now().astimezone().isoformat()
            after = read_robot_state(receiver)

            if not ok:
                print("Image capture failed.")
                continue

            gray_saved = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected, _ = cv2.findChessboardCorners(
                gray_saved, board_size, None
            )

            if not detected:
                print("Saved-frame checkerboard detection failed.")
                continue

            # Reject captures if the joints were moving.
            velocities = (
                before["joint_velocities_rad_s"]
                + after["joint_velocities_rad_s"]
            )
            maximum_speed = max(abs(v) for v in velocities)

            changes = [
                abs(a - b)
                for a, b in zip(
                    before["joint_positions_rad"],
                    after["joint_positions_rad"]
                )
            ]
            maximum_change = max(changes)

            if maximum_speed > 0.01 or maximum_change > 0.001:
                print("Robot moved during acquisition. Try again.")
                continue

            # Give the image and metadata the same unique ID.
            observation_id = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            image_path = output_dir / f"{observation_id}.png"
            metadata_path = output_dir / f"{observation_id}.json"

            metadata = {
                "observation_id": observation_id,
                "image_filename": image_path.name,
                "image_time_host_approx": image_time,
                "camera_index": camera_index,
                "board_size": board_size,
                "square_size_mm": square_size_mm,
                "intrinsics_file":
                    "configs/lab_camera_01_intrinsics_23p5mm_preliminary.npz",
                "robot_before": before,
                "robot_after": after,
                "maximum_joint_change_rad": maximum_change
            }

            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError("Failed to save image")

            with metadata_path.open("x", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2)

            print("Image saved:", image_path)
            print("Robot data saved:", metadata_path)


finally:
    camera.release()

    if receiver is not None:
        receiver.disconnect()

    cv2.destroyAllWindows()
