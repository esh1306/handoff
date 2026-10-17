"""Synthetic track.csv generator: reach -> grasp -> lift 15cm -> move left 20cm -> lower -> release.

Lets actions.py, vlm.py, export.py and replay.py be built and tested before
perception.py produces real tracks.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRACK_COLUMNS = [
    "frame", "t",
    "cam_x", "cam_y", "cam_z",
    "tbl_x", "tbl_y", "tbl_z",
    "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
    "grip", "hand_conf", "occluded", "handedness",
]

# (t_start, t_end, tbl_xyz_start, tbl_xyz_end, grip_start, grip_end)
PHASES = [
    (0.0, 4.0, (0.30, 0.00, 0.20), (0.35, 0.00, 0.05), 0, 0),   # reach
    (4.0, 4.5, (0.35, 0.00, 0.05), (0.35, 0.00, 0.05), 0, 1),   # grasp
    (4.5, 8.5, (0.35, 0.00, 0.05), (0.35, 0.00, 0.20), 1, 1),   # lift 15cm
    (8.5, 12.5, (0.35, 0.00, 0.20), (0.35, 0.20, 0.20), 1, 1),  # move left 20cm
    (12.5, 16.5, (0.35, 0.20, 0.20), (0.35, 0.20, 0.05), 1, 1),  # lower
    (16.5, 17.0, (0.35, 0.20, 0.05), (0.35, 0.20, 0.05), 1, 0),  # release
    (17.0, 20.0, (0.35, 0.20, 0.05), (0.35, 0.20, 0.05), 0, 0),  # hold
]

IDENTITY_ROT6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _smoothstep(u):
    return u * u * (3 - 2 * u)


def _phase_at(t):
    for t_start, t_end, p_start, p_end, grip_start, grip_end in PHASES:
        if t_start <= t < t_end:
            dur = t_end - t_start
            u = 0.0 if dur <= 0 else (t - t_start) / dur
            pos = tuple(
                a + (b - a) * _smoothstep(u) for a, b in zip(p_start, p_end)
            )
            grip = grip_start if u < 0.5 else grip_end
            return pos, grip
    raise ValueError(f"t={t} outside all phases")


def generate_synth_track(fps=30):
    """Return a DataFrame matching the track.csv data contract."""
    duration = PHASES[-1][1]
    n_frames = int(round(duration * fps))
    rows = []
    for frame in range(n_frames):
        t = frame / fps
        (tbl_x, tbl_y, tbl_z), grip = _phase_at(t)

        # Rough synthetic camera extrinsic: table forward -> +z_cam, table
        # left -> -x_cam, table up -> -y_cam (EgoLAP human convention),
        # with the table origin ~0.6m in front of, and 0.4m below, the camera.
        cam_x = -tbl_y
        cam_y = -tbl_z + 0.4
        cam_z = tbl_x + 0.6

        rows.append([
            frame, t,
            cam_x, cam_y, cam_z,
            tbl_x, tbl_y, tbl_z,
            *IDENTITY_ROT6D,
            grip, 1.0, 0, "Right",
        ])

    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", default="takes/synth", help="take directory to write track.csv into")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.take)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "track.csv"

    df = generate_synth_track(fps=args.fps)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} frames to {out_path}")


if __name__ == "__main__":
    main()
