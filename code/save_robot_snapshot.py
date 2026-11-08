
from pathlib import Path
from datetime import datetime
import json

from rtde_receive import RTDEReceiveInterface

# Address of the laboratory robot controller.
robot_ip = "192.168.100.110"

# Find the thesis project directory.
project_root = Path(__file__).resolve().parent.parent

# Connect using the read-only RTDE interface.
receiver = RTDEReceiveInterface(robot_ip)

try:
    # Record the approximate time of acquisition.
    capture_time = datetime.now().astimezone()

    # Collect the robot's controller-reported state.
    snapshot = {
        "capture_time": capture_time.isoformat(),
        "robot_ip": robot_ip,
        "controller_timestamp_s": receiver.getTimestamp(),
        "actual_joint_positions_rad": receiver.getActualQ(),
        "actual_joint_velocities_rad_s": receiver.getActualQd(),
        "actual_tcp_pose": receiver.getActualTCPPose()
    }

finally:
    # Always close the network connection.
    receiver.disconnect()

# Prepare a directory for raw robot measurements.
output_dir = project_root / "data/raw/robot_01/state_snapshots"
output_dir.mkdir(parents=True, exist_ok=True)

# Give each snapshot a timestamped filename.
filename = f"robot_state_{capture_time.strftime('%Y%m%d_%H%M%S_%f')}.json"
output_file = output_dir / filename

# Save the measurements in JSON format.
with output_file.open("x", encoding="utf-8") as file:
    json.dump(snapshot, file, indent=2)

print("Robot snapshot saved to:", output_file)
