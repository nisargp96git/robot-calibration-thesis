import cv2
import time
# Import Path for creating and managing file and directory paths.
from pathlib import Path
#open camera index 4 (dev/video4)
#ls -l /dev/video*
#v4l2-ctl --list-devices
camera = cv2.VideoCapture(4, cv2.CAP_V4L2)



if not camera.isOpened():
    print("Error: Could not open the Camera.")
    exit()

print("Camera opened Successfully.")

# Request MJPG (Motion JPEG) video encoding from the webcam.
# MJPG reduces USB bandwidth requirements compared with uncompressed video.

camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("Camera resolution:", width, "x", height)


# Capture one frame (image) from the webcam.
# camera.read() returns two values:
# ret   = True if a frame was captured successfully, otherwise False.
# frame = The captured image stored as a NumPy array.
for frame_number in range(1,11):
      
    ret, frame = camera.read()

    if not ret:

        print("Error: Could not capture an image.")
        break


# Display the dimensions of the captured image.
# frame.shape returns (height, width, number_of_channels).
    print("Captured image shape:", frame.shape)

    time.sleep(0.1)

# Define the directory where raw intrinsic-calibration images are stored.
output_dir = Path("data/raw/camera_01/intrinsic")

# Create the directory and all missing parent directories.
# parents=True creates intermediate directories if necessary.
# exist_ok=True prevents errors if the directory already exists.
output_dir.mkdir(parents=True, exist_ok=True)

# Start searching for an available image number from 1
image_number = 1

#Generate the filename using a three digit number (001, 002, 003...)
image_path = output_dir / f"image_{image_number:03d}.png"

# Keep Searching while a file with the current name already exist
while image_path.exists():

      image_number += 1
      image_path = output_dir / f"image_{image_number:03d}.png"

# Save the captured image as a PNG file.
# imwrite() returns True if the image was successfully saved.
saved = cv2.imwrite(str(image_path), frame)

# Check whether the image was saved successfully.
if saved:
        print("Image saved successfully:", image_path)
else:
        print("Error: Image could not be saved.")

camera.release()