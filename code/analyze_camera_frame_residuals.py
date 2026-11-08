
from pathlib import Path
import json

import numpy as np

# Locate the saved calibration and validation results.
root = Path(__file__).resolve().parent.parent
directory = root / "data/processed/eye_to_hand/run_01"

# Load the camera orientation relative to the robot base.
with np.load(directory / "eye_to_hand_initial.npz") as data:
    R_B_C = data["T_base_camera"][:3, :3]

# Load translation residuals expressed in robot-base coordinates.
with (directory / "eye_to_hand_validation.json").open(
    "r", encoding="utf-8"
) as file:
    report = json.load(file)

E_B = np.array(
    [item["translation_components_mm"] for item in report["results"]],
    dtype=np.float64
)

# Express the same residual vectors in camera coordinates.
E_C = E_B @ R_B_C

# Rotation must preserve the length of each residual vector.
if not np.allclose(
    np.linalg.norm(E_B, axis=1),
    np.linalg.norm(E_C, axis=1),
    atol=1e-6
):
    raise RuntimeError("Coordinate transformation check failed.")

print("RESIDUAL COMPONENTS IN CAMERA COORDINATES")
print("Observation | X (mm) | Y (mm) | Z-depth (mm)")

for item, error in zip(report["results"], E_C):
    print(
        f"{item['observation_id']} | "
        f"{error[0]:+.4f} | "
        f"{error[1]:+.4f} | "
        f"{error[2]:+.4f}"
    )

rms = np.sqrt(np.mean(E_C**2, axis=0))

print("\nCAMERA-COORDINATE RMS")
print(f"X RMS (mm): {rms[0]:.6f}")
print(f"Y RMS (mm): {rms[1]:.6f}")
print(f"Z-depth RMS (mm): {rms[2]:.6f}")

print(
    "Overall 3D RMS (mm):",
    np.sqrt(np.mean(np.sum(E_C**2, axis=1)))
)