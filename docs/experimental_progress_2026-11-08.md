# Experimental Progress and Results Log

**Project:** Master's Thesis — UR3e Robot Repeatability Using Low-Cost Eye-to-Hand Vision Calibration
**Institution:** Technische Hochschule Deggendorf
**Report date:** 8 November 2026
**Thesis submission deadline:** 28 January 2027
**Status:** Initial single-camera calibration, validation and stationary repeatability characterization completed.

## 1. Research objective

The primary research objective is to investigate the physical positioning repeatability of a Universal Robots UR3e robot using a low-cost camera-based measurement system.

The research also investigates:

- The quality and consistency of the eye-to-hand calibration.
- The measurement repeatability and limitations of low-cost webcams.
- The effect of robot configuration and payload on positioning repeatability.
- The potential improvement provided by multiple cameras.
- The feasibility of AI-based modelling or compensation of systematic pose-dependent deviations.

Whether absolute robot positioning accuracy is also included remains subject to confirmation with the thesis supervisor.

## 2. Experimental setup

| Component | Description |
|---|---|
| Robot | Universal Robots UR3e |
| Camera arrangement | Stationary eye-to-hand |
| Camera | Low-cost USB webcam, manually adjustable focus |
| Image resolution | 1920 × 1080 pixels |
| Calibration target | Planar checkerboard |
| Detected pattern | 9 × 6 internal corners (54 total) |
| Nominal modelled corner pitch | 23.5 mm |
| Operating system | Ubuntu 24.04 |
| Programming language | Python 3.12 |
| OpenCV | 4.13.0 |
| NumPy | 2.5.3 |
| Robot communication | RTDE |
| Data storage | Local thesis project; raw datasets excluded from Git |

**Checkerboard geometry note:** Physical measurements previously indicated approximately 23.556 mm horizontal pitch and 23.500 mm vertical pitch. Current analysis uses a uniform 23.5 mm pitch. The possible effect of this approximation requires further investigation.

## 3. Camera intrinsic calibration

**Dataset:** 26 intrinsic calibration images and 9 separate validation images.

The checkerboard corners were detected using OpenCV and refined to subpixel precision. The intrinsic camera matrix and lens distortion coefficients were estimated using the calibration images.

| Metric | Result |
|---|---:|
| Calibration reprojection RMS | 0.2857 pixels |
| Independent validation reprojection RMS | 0.3017 pixels |
| Calibration images | 26 |
| Validation images | 9 |

**Saved calibration:**
`configs/lab_camera_01_intrinsics_23p5mm_preliminary.npz`

The results quantify image reprojection consistency and do not independently establish metric measurement accuracy.

## 4. Eye-to-hand calibration

**Acquisition date:** 5 November 2026.

The checkerboard was rigidly attached to the robot flange. The camera remained stationary. At each observation, an image and robot-controller state were recorded.

**Dataset:**

- 30 paired camera–robot observations.
- One repeated robot configuration excluded.
- 23 observations used for calibration.
- 6 observations reserved for validation.

The geometric relationship was:

`T_B_F(i) × T_F_P = T_B_C × T_C_P(i)`

Where:

- `T_B_F` is the robot-controller-derived base-to-flange transformation.
- `T_C_P` is the camera-to-checkerboard transformation estimated from the image.
- `T_B_C` is the unknown base-to-camera transformation.
- `T_F_P` is the unknown flange-to-checkerboard transformation.

The fixed transformations were estimated using OpenCV hand-eye calibration and the recorded observations.

### Estimated camera-to-base translation

`[-1185.254, -48.420, 635.593] mm`

### Estimated flange-to-checkerboard translation

`[103.562, -57.083, 26.026] mm`

### Calibration and validation results

| Metric | Training | Held-out validation |
|---|---:|---:|
| Observations | 23 | 6 |
| Translation RMS consistency | 1.6228 mm | 1.3429 mm |
| Rotation RMS consistency | 0.1816° | 0.1860° |
| Maximum translation residual | 3.1487 mm | 1.9994 mm |

These results describe internal robot–camera geometric consistency, not independently verified absolute positioning accuracy or robot return-to-pose repeatability.

**Important files:**

- `code/capture_eye_to_hand.py`
- `code/audit_eye_to_hand.py`
- `code/estimate_board_poses.py`
- `code/calibrate_eye_to_hand.py`
- `code/validate_eye_to_hand.py`
- `data/processed/eye_to_hand/run_01/eye_to_hand_initial.npz`
- `data/processed/eye_to_hand/run_01/eye_to_hand_validation.json`

## 5. Stationary repeatability experiments

Two types of stationary measurement experiments were performed:

1. Checkerboard attached to the robot, with the robot nominally holding a fixed configuration.
2. Checkerboard mounted independently of the robot, with the board stationary.

Each experiment included 30 images at near, middle and far configurations.

### Three-dimensional RMS positional variation

| Position | Robot-mounted board | Independently fixed board |
|---|---:|---:|
| Near | 0.0924 mm | 0.1122 mm |
| Middle | 0.1241 mm | 0.0714 mm |
| Far | 0.3250 mm | 0.2814 mm |

### Fixed-board rotation RMS variation

| Position | Rotation RMS |
|---|---:|
| Near | 0.0361° |
| Middle | 0.0685° |
| Far | 0.0984° |

### Fixed-board mean target depth

| Position | Camera Z |
|---|---:|
| Near | 691.45 mm |
| Middle | 935.37 mm |
| Far | 1329.33 mm |

These experiments characterize short-term pose-estimation variation. They do not yet measure how consistently the UR3e returns to a commanded pose after moving away.

The robot-mounted and independently fixed board configurations also differed in target distance and orientation, limiting direct causal comparison.

## 6. Eye-to-hand residual analysis

Six held-out validation observations were analyzed in robot-base and camera coordinates.

### Residual RMS in robot-base coordinates

| Axis | RMS |
|---|---:|
| X | 1.1690 mm |
| Y | 0.4740 mm |
| Z | 0.4608 mm |

### Residual RMS in camera coordinates

| Axis | RMS |
|---|---:|
| X | 0.4754 mm |
| Y | 0.6456 mm |
| Z (optical depth) | 1.0773 mm |

**Overall 3D RMS:** 1.3429 mm in either coordinate frame.

The optical-depth component contributes approximately 64% of the squared three-dimensional validation discrepancy.

This motivates further investigation of depth-related measurement effects, robot kinematics and systematic calibration errors. It does not prove that the camera alone causes the residual.

## 7. Current limitations

- The robot's actual return-to-pose repeatability has not yet been experimentally evaluated using repeated movement cycles.
- Absolute positioning accuracy has not been independently measured.
- Short-term camera measurement variation may be too large to resolve very small UR3e positioning variations.
- The effect of physical checkerboard geometry deviations has not yet been quantified.
- Eye-to-hand validation used six held-out observations from the same acquisition session.
- Multi-camera calibration and fusion have not yet been implemented.
- Payload-dependent and configuration-dependent robot repeatability experiments remain outstanding.
- AI-based modelling and compensation have not yet been implemented.

## 8. Planned next experiments

1. Design and implement repeated UR3e return-to-pose measurements.
2. Capture multiple camera frames per robot arrival to characterize within-arrival and between-arrival variation.
3. Compare repeatability across robot configurations, approach conditions and approved payloads.
4. Calibrate a second camera and evaluate multi-view measurement capability.
5. Assess the feasibility of AI-based modelling using properly defined experimental targets and independent evaluation data.

## 9. Reproducibility and data management

The source code, documentation and environment dependencies should be version-controlled with Git.

Raw image datasets and robot acquisition records are retained separately from the code repository. They must be backed up independently because excluding files from Git does not protect them from local data loss.

Changes to calibration assumptions, camera focus, target geometry or software versions must be documented before results from different experiments are compared.

**Next project milestone:** Establish the UR3e return-to-pose experimental procedure and collect the first repeated movement dataset.