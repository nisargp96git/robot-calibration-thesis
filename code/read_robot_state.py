
import sys
from rtde_receive import RTDEReceiveInterface

# Require the robot's IP address as a command-line argument.
if len(sys.argv) != 2:
    raise SystemExit("Usage: python code/read_robot_state.py ROBOT_IP")

# Read the IP address supplied in the terminal.
robot_ip = sys.argv[1]

# Establish a read-only RTDE connection.
receiver = RTDEReceiveInterface(robot_ip)

try:
    # Read the six actual joint positions, in radians.
    joint_positions = receiver.getActualQ()

    # Read the six actual joint velocities, in radians/second.
    joint_velocities = receiver.getActualQd()

    # Read the controller-reported TCP pose.
    tcp_pose = receiver.getActualTCPPose()

    # Display the received data.
    print("Actual joint positions (rad):", joint_positions)
    print("Actual joint velocities (rad/s):", joint_velocities)
    print("TCP position (m):", tcp_pose[:3])
    print("TCP orientation (rotation vector, rad):", tcp_pose[3:])

finally:
    # Close the RTDE connection, including if a read fails.
    receiver.disconnect()
