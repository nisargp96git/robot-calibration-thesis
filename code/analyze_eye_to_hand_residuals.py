
from pathlib import Path
import json

import numpy as np


# Locate the previously saved validation results.
project_root = Path(__file__).resolve().parent.parent

validation_path = (
    project_root
    / "data/processed/eye_to_hand/run_01"
    / "eye_to_hand_validation.json"
)


# Load the six held-out observations.
with validation_path.open("r", encoding="utf-8") as file:
    report = json.load(file)

observations = report["results"]

if len(observations) != 6:
    raise RuntimeError("Expected six validation observations.")


# Extract X, Y and Z translation residuals.
errors = np.array(
    [
        observation["translation_components_mm"]
        for observation in observations
    ],
    dtype=np.float64
)


# Print the individual error vectors.
print("INDIVIDUAL VALIDATION RESIDUALS")
print("Observation | X (mm) | Y (mm) | Z (mm) | 3D (mm)")

for observation, error in zip(observations, errors):
    magnitude = np.linalg.norm(error)

    print(
        f"{observation['observation_id']} | "
        f"{error[0]:+.4f} | "
        f"{error[1]:+.4f} | "
        f"{error[2]:+.4f} | "
        f"{magnitude:.4f}"
    )


# Calculate statistics for each robot-base axis.
print("\nAXIS-BY-AXIS SUMMARY")

for axis_index, axis_name in enumerate(("X", "Y", "Z")):
    axis_errors = errors[:, axis_index]

    mean_error = float(np.mean(axis_errors))

    rms_error = float(
        np.sqrt(np.mean(axis_errors**2))
    )

    std_error = float(
        np.std(axis_errors, ddof=1)
    )

    print(f"\n{axis_name} axis:")
    print(f"  Mean signed residual (mm): {mean_error:+.6f}")
    print(f"  RMS residual (mm): {rms_error:.6f}")
    print(f"  Standard deviation (mm): {std_error:.6f}")


# Check the calculated 3D RMS against Step 91.
rms_3d = float(
    np.sqrt(np.mean(np.sum(errors**2, axis=1)))
)

saved_rms = float(report["translation_rms_mm"])

if not np.isclose(rms_3d, saved_rms, atol=1e-6):
    raise RuntimeError("3D RMS does not match Step 91.")

print("\nOVERALL VALIDATION")
print(f"3D translation RMS (mm): {rms_3d:.6f}")
print("Matches Step 91:", True)
