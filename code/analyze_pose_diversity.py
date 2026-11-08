
from pathlib import Path
import json

import cv2
import numpy as np


project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data/raw/eye_to_hand/run_01"

records = []

# Load every recorded robot flange pose.
for json_path in sorted(data_dir.glob("*.json")):
    with json_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    pose = np.asarray(
        metadata["robot_after"]["flange_pose_m_rad"],
        dtype=np.float64,
    )

    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"Invalid flange pose: {json_path.name}")

    position_mm = pose[:3] * 1000.0
    rotation_matrix, _ = cv2.Rodrigues(pose[3:])

    records.append({
        "id": metadata["observation_id"],
        "position_mm": position_mm,
        "rotation_matrix": rotation_matrix,
    })

if len(records) < 3:
    raise RuntimeError("At least three observations are required")

# Analyze the range of flange positions.
positions = np.array([r["position_mm"] for r in records])
position_ranges = np.ptp(positions, axis=0)

print("OBSERVATIONS:", len(records))
print("Position ranges along base X, Y, Z (mm):")
print(np.round(position_ranges, 2))

# Compare every pair of recorded flange poses.
pairs = []

for i in range(len(records)):
    for j in range(i + 1, len(records)):
        first = records[i]
        second = records[j]

        distance_mm = np.linalg.norm(
            first["position_mm"] - second["position_mm"]
        )

        relative_rotation = (
            first["rotation_matrix"].T
            @ second["rotation_matrix"]
        )

        cosine = (np.trace(relative_rotation) - 1.0) / 2.0
        angle_deg = np.degrees(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )

        pairs.append((
            first["id"],
            second["id"],
            float(distance_mm),
            float(angle_deg),
        ))

distances = np.array([pair[2] for pair in pairs])
angles = np.array([pair[3] for pair in pairs])

print("\nPAIRWISE POSITION DIFFERENCES (mm)")
print("Minimum:", np.min(distances))
print("Median:", np.median(distances))
print("Maximum:", np.max(distances))

print("\nPAIRWISE ROTATION DIFFERENCES (degrees)")
print("Minimum:", np.min(angles))
print("Median:", np.median(angles))
print("Maximum:", np.max(angles))

print("\nPOSSIBLE DUPLICATE POSES")
duplicates = [
    pair for pair in pairs
    if pair[2] < 2.0 and pair[3] < 2.0
]

if not duplicates:
    print("No near-duplicates found")
else:
    for first_id, second_id, distance, angle in duplicates:
        print(
            first_id, second_id,
            f"distance={distance:.3f} mm",
            f"rotation={angle:.3f} deg",
        )

# Examine rotation variation along independent directions.
reference_rotation = records[0]["rotation_matrix"]

rotation_vectors = np.array([
    cv2.Rodrigues(
        record["rotation_matrix"] @ reference_rotation.T
    )[0].reshape(3)
    for record in records
])

centered = rotation_vectors - rotation_vectors.mean(axis=0)
singular_values = np.linalg.svd(
    centered, compute_uv=False
)

print("\nROTATION DIVERSITY — SINGULAR VALUES")
print(np.round(singular_values, 4))
