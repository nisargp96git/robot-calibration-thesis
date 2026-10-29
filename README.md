# Robot Calibration Thesis

## Overview

This repository contains the software, documentation, and development history for my master's thesis on robot calibration.

Raw experimental datasets are stored locally in `data/raw/` and are excluded from Git version control. They require separate backups.

The project is being developed incrementally, with the setup, commands, code, experiments, results, and troubleshooting documented throughout the development process.

## Repository Structure

- `docs/` — Setup notes, theory, experiments, and troubleshooting
- `code/` — Source code for camera, robot, and calibration tasks
- `data/` — Experimental and calibration data
- `cad/` — CAD and SolidWorks-related files
- `configs/` — Configuration files
- `references/` — Reference-related project material

## Project Status

**Current Phase:** Intrinsic Camera Calibration - Preliminary Results

### Completed

- Python virtual environment configured on Ubuntu
- OpenCV and NumPy installed.
- USB webcam acquisition configured at 1920 x 1080 using MJPG and V4L2.
- Live camera preview implemented in `code/live_capture.py`.
- Real-time detection of a 9 × 6 checkerboard pattern (54 internal corners).
- Subpixel corner refinement implemented using OpenCV.
- Keyboard-controlled image acquisition with sequential PNG filenames.
- Initial intrinsic camera calibration using 7 images.
- Per-image reprojection error analysis completed.

### Preliminary Intrinsic Calibration Results

- Image resolution: 1920 × 1080 pixels
- Checkerboard square size: 25 mm
- Internal corners: 9 × 6 (54 points)
- Calibration images: 7
- Total corner observations: 378
- Overall RMS reprojection error: 0.277096 pixels

These results are preliminary and have not yet been independently validated.

### In Progress / Planned

- Expand the intrinsic calibration dataset with diverse checkerboard views.
- Evaluate calibration parameter stability and independent reprojection accuracy.
- Save calibrated camera parameters for reuse.
- Perform eye-to-hand extrinsic camera calibration.
- Establish the external robot measurement reference frame.
- Collect baseline robot positioning errors under different poses and loads.
- Investigate robot calibration and AI-based residual error compensation.
- Compare baseline and corrected positioning accuracy.