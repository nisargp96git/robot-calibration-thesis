import cv2
import time
from pathlib import Path

#v4l2-ctl --list-devices

camera = cv2.VideoCapture(2, cv2.CAP_V4L2)

if not camera.isOpened():
    raise RuntimeError("Error: Could not open the webcam.")

 #Request MJPG conversion for transferring images via usb.

camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

#Requesting HD resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Defining the checkerboard pattern
board_size = (9,6)


#Define when the subpixel corner refinement should stop.

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


#Saving the captured images in the correct folder
#Preserving the previous seven calibration images.
project_root = Path(__file__).resolve().parent.parent

#for camera 1 lab main folder for calibration
Output_dir = project_root /"data/raw/camera_01/intrinsic/final_run_01"

#for camera 1 lab validation folder, not for intrinsics
#Output_dir = project_root / "data/raw/camera_01/intrinsic/validation_run_01"




Output_dir.mkdir(parents=True, exist_ok=True)

try:
    for frame_number in range(1,11):
        ret, frame = camera.read()

        if not ret:
            raise RuntimeError("Camera initialization failed.")

        time.sleep(0.1)
    #Confirm completion of camera warm-up.
    print("Camera initialized. Live preview starting.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error: Failed to capture a frame.")
            break

        #Display the current frame in a Window.
        #cv2.imshow("Camera 01 - Live Preview", frame)

        #Converting the captured BGR image to grayscale image.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Checking for the checkerboard corner pattern
        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        preview = frame.copy()

        if found:

            #Improve the initial corner estimates to subpixel precison
            corners= cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )
            #Draw the refined corner locations on the preview image.

            cv2.drawChessboardCorners(preview, board_size, corners, found)

            status_text = "CHECKERBOARD DETECTED: 54/54"
            status_color = (0,255,0)

        else:
            status_text = "CHECKERBOARD NOT DETECTED"
            status_color = (0,0,255)

        cv2.putText(
            preview, status_text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA
        )

        
        cv2.imshow("Camera 01 - Live Preview", preview)

        #Process window events and read keyboard inputs./ 0xFF retains the lowest bit of the keyboard code.
        key= cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if found:
                image_number = 1
                image_path = Output_dir / f"image_{image_number:03d}.png"

                while image_path.exists():
                    image_number +=1
                    image_path= Output_dir / f"image_{image_number:03d}.png"

                saved = cv2.imwrite(str(image_path), frame)

                if saved:
                    print("Image Saved:", image_path)

                else:
                    print("Error: Failed to save image")

            else:
                print("Image not saved. Checkerboard not fully detected")

        if key == ord("q"):
            break

finally:
    camera.release()

    cv2.destroyAllWindows()