
from rtde import rtde

# Robot controller connection details.
robot_ip = "192.168.100.110"
robot_port = 30004

# Fields we want to read from the robot.
fields = [
    "timestamp",
    "tcp_offset",
    "actual_TCP_pose",
    "payload",
    "payload_cog"
]

# RTDE data types corresponding to those fields.
types = [
    "DOUBLE",
    "VECTOR6D",
    "VECTOR6D",
    "DOUBLE",
    "VECTOR3D"
]

# Create the RTDE connection.
connection = rtde.RTDE(robot_ip, robot_port)

try:
    # Establish the connection to the robot.
    connection.connect()

# Check whether the connection is open.
    if not connection.is_connected():
        raise ConnectionError("Could not connect to the robot")

    print("Controller version:", connection.get_controller_version())

    # Configure the fields to be received at 10 Hz.
    if not connection.send_output_setup(fields, types, frequency=10):
        raise RuntimeError("RTDE output setup failed")

    # Start receiving data; no robot motion is commanded.
    if not connection.send_start():
        raise RuntimeError("RTDE synchronization failed")

    # Receive one robot-state sample.
    state = connection.receive()

    if state is None:
        raise RuntimeError("No robot data received")

    # Display the active robot configuration.
    print("Controller timestamp (s):", state.timestamp)
    print("Active TCP offset:", state.tcp_offset)
    print("Actual TCP pose:", state.actual_TCP_pose)
    print("Active payload (kg):", state.payload)
    print("Payload CoG (m):", state.payload_cog)

finally:
    connection.disconnect()
