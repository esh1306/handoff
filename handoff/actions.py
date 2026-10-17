"""track.csv -> actions.jsonl: per-window motion deltas rendered as EgoLAP language actions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HAND_CONF_THRESHOLD = 0.5


def rot6d_to_matrix(rot6d):
    """Gram-Schmidt reconstruction of a rotation matrix from its first two columns."""
    a1 = np.array(rot6d[0:3], dtype=float)
    a2 = np.array(rot6d[3:6], dtype=float)
    x = a1 / np.linalg.norm(a1)
    y = a2 - np.dot(a2, x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def round_cm(delta_m):
    return int(round(delta_m * 100))


def round_deg5(delta_deg):
    return int(round(delta_deg / 5.0) * 5)


def format_action_text(dx_cm, dy_cm, dz_cm, roll_deg, pitch_deg, yaw_deg, grip_end, handedness):
    """Render one EgoLAP clause. Order: x, z, y, roll, pitch, yaw, gripper. Zero components omitted."""
    clauses = []
    if dx_cm != 0:
        clauses.append(f"move {'forward' if dx_cm > 0 else 'back'} {abs(dx_cm)} cm")
    if dz_cm != 0:
        clauses.append(f"move {'up' if dz_cm > 0 else 'down'} {abs(dz_cm)} cm")
    if dy_cm != 0:
        clauses.append(f"move {'left' if dy_cm > 0 else 'right'} {abs(dy_cm)} cm")
    if roll_deg != 0:
        clauses.append(f"tilt {'left' if roll_deg > 0 else 'right'} {abs(roll_deg)} degrees")
    if pitch_deg != 0:
        clauses.append(f"tilt {'back' if pitch_deg > 0 else 'forward'} {abs(pitch_deg)} degrees")
    if yaw_deg != 0:
        clauses.append(f"rotate {'counterclockwise' if yaw_deg > 0 else 'clockwise'} {abs(yaw_deg)} degrees")
    clauses.append(f"{grip_end} gripper")

    prefix = "Right arm: " if handedness == "Right" else "Left arm: "
    return prefix + ", ".join(clauses)


def windows_from_track(df, window_s, fps):
    duration = df["t"].max() + 1.0 / fps
    n_windows = int(np.ceil(duration / window_s))
    for w in range(n_windows):
        t_start = w * window_s
        t_end = t_start + window_s
        window_df = df[(df["t"] >= t_start) & (df["t"] < t_end)]
        if len(window_df) == 0:
            continue
        yield w, t_start, t_end, window_df


def _endpoints(window_df):
    """First/last non-occluded row in the window; falls back to all rows if every row is occluded."""
    valid = window_df[window_df["occluded"] == 0]
    source = valid if len(valid) > 0 else window_df
    return source.iloc[0], source.iloc[-1], len(valid) == 0


def compute_window_action(window_df, window, t_start, t_end):
    first, last, fully_occluded = _endpoints(window_df)
    occluded_frac = float((window_df["occluded"] == 1).mean())

    dx_cm = round_cm(last["tbl_x"] - first["tbl_x"])
    dy_cm = round_cm(last["tbl_y"] - first["tbl_y"])
    dz_cm = round_cm(last["tbl_z"] - first["tbl_z"])

    rot_valid = bool(
        not fully_occluded
        and first["hand_conf"] > HAND_CONF_THRESHOLD
        and last["hand_conf"] > HAND_CONF_THRESHOLD
    )
    if rot_valid:
        r_start = rot6d_to_matrix([first[f"rot6d_{i}"] for i in range(6)])
        r_end = rot6d_to_matrix([last[f"rot6d_{i}"] for i in range(6)])
        r_rel = r_end @ r_start.T
        roll, pitch, yaw = Rotation.from_matrix(r_rel).as_euler("xyz", degrees=True)
        roll_deg, pitch_deg, yaw_deg = round_deg5(roll), round_deg5(pitch), round_deg5(yaw)
    else:
        roll_deg = pitch_deg = yaw_deg = 0

    grip_end = "close" if last["grip"] == 1 else "open"
    handedness = last["handedness"]

    # Camera-frame translation: forward = +z_cam, left = -x_cam, up = -y_cam.
    cam_dx_cm = round_cm(last["cam_z"] - first["cam_z"])
    cam_dy_cm = round_cm(-(last["cam_x"] - first["cam_x"]))
    cam_dz_cm = round_cm(-(last["cam_y"] - first["cam_y"]))

    text = format_action_text(dx_cm, dy_cm, dz_cm, roll_deg, pitch_deg, yaw_deg, grip_end, handedness)
    text_camera_frame = format_action_text(cam_dx_cm, cam_dy_cm, cam_dz_cm, 0, 0, 0, grip_end, handedness)

    return {
        "window": window,
        "t_start": t_start,
        "t_end": t_end,
        "frame": "table",
        "dx_cm": dx_cm, "dy_cm": dy_cm, "dz_cm": dz_cm,
        "roll_deg": roll_deg, "pitch_deg": pitch_deg, "yaw_deg": yaw_deg,
        "grip_end": grip_end,
        "rot_valid": rot_valid,
        "occluded_frac": occluded_frac,
        "text": text,
        "text_camera_frame": text_camera_frame,
    }


def build_actions(track_csv, window_s=1.0, fps=30):
    df = pd.read_csv(track_csv)
    return [
        compute_window_action(window_df, w, t_start, t_end)
        for w, t_start, t_end, window_df in windows_from_track(df, window_s, fps)
    ]


def write_actions_jsonl(take_dir, window_s=1.0, fps=30):
    take_dir = Path(take_dir)
    actions = build_actions(take_dir / "track.csv", window_s=window_s, fps=fps)
    out_path = take_dir / "actions.jsonl"
    with open(out_path, "w") as f:
        for action in actions:
            f.write(json.dumps(action) + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take", help="take directory containing track.csv")
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    out_path = write_actions_jsonl(args.take, window_s=args.window_s, fps=args.fps)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
