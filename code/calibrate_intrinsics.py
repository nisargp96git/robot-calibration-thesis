import cv2
from pathlib import Path
import numpy as np


project_root= Path(__file__).resolve().parent.parent

image_dir = project_root / "data/raw/camera_01/intrinsic"

#sorted() arranges the files in filename order.
image_paths = sorted(image_dir.glob("image_*.png"))
print("Images found:", len(image_paths))

#Define the number of internal corners in the checkerboard
board_size = (9, 6)

#Defining the  stopping condition for stopping condition for  subpixel corner refinement

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# Use one checkerboard square width as the coordinate
square_size = 25.0

#Create an empty list to store the 3D coordinates 
board_object_points = []

#loop through all 6 rows of internal corners.
for row in range(board_size[1]):
    #loop through all the 9 columns in current row.
    for col in range(board_size[0]):

        #store X, Y, Z positions of the current corner.
        board_object_points.append([
            col * square_size,
            row * square_size,
            0.0
        ])
#Convert the list into a Numpy array of 32-bit floating point
board_object_points = np.asarray(
    board_object_points,
    dtype = np.float32
)
#Display the dimension of the resulting array.
print("3D reference point shape:", board_object_points.shape)

#Display the first 10 reference points.
print("First 10 reference points:")
print(board_object_points[:10])

#create and empty list to store the 3d checkerboard
object_points = []

#create an empty list to st0re the detected 2D image points
image_points = []

#Processing each discovered image.

for image_path in image_paths:
    #Load the image from disk as a NumPy array.
    image = cv2.imread(str(image_path))

    #Check whether OpenCV failed to load the image.
    if image is None:
        print("Error loading:", image_path.name)
        continue

     #Extract the image height and width from its array shape.
    height, width = image.shape[:2]

    #print the file name and image dimensions
    print(f"{image_path.name}: {width} x {height}")

    #convert the BGR color image into a grayscale image.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Detect the complete checkerboard pattern
    found, corners = cv2.findChessboardCorners(
        gray,
        board_size,
        None
    )
    #Check whether the entire checkerboard pattern is detected.
    if found:

          
        
        #Improve the initial corner estimates to subpixel precison
        corners= cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria
        )

        number_of_corners = len(corners)
        #Display the number of successfully detected corners.
        print(f" Checkerboard detected: {number_of_corners} corners")
        #Display the dimensions of the detected corner coordinate array.
        print("corner array shape:", corners.shape)
        #Display the refined coordinates of the first detected corner.
        print("First refined corner (u, v):", corners[0])

        #Store a copy of the 54 physical checkerboard points
        object_points.append (board_object_points.copy())

        #store a copy of the 54 refined 2d corner coordinates
        image_points.append(corners.copy())

    else:
        print("checkerboard NOT detected.")
#Display the number of images successfully prepared for calibration
print("Calibration images accepted:", len(image_points))

#Only inspect the 1st image if the last image was accepted
if image_points:
    #Display the shape of the 1st image if atleast one image was accepted.
    print("3D points shape:", object_points[0].shape)
    #Display the shape of the 1st images 2d coordinate 
    print("2D Points shape:", image_points[0].shape)

    #Define the image resolution as (width, height).
    image_size = (1920, 1080)

    #perform intrinsic camera calibration using all accepted images.
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    #Display the overall root-mean-square reprojection error.
    print("Calibration rms error;", rms)

    #Display the estimated 3x3 intrinsic camera matrix.
    print("camera matrix:")
    print(camera_matrix)

    #Display the estimated lens distortion coefficients.
    print("Distortion Coefficients")
    print(dist_coeffs)
