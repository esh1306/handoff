import numpy as np
import pandas as pd
import pytest

from handoff.actions import (
    build_actions,
    compute_window_action,
    format_action_text,
    rot6d_to_matrix,
    round_cm,
    round_deg5,
)
from handoff.synth import generate_synth_track

TRACK_COLUMNS = [
    "t", "tbl_x", "tbl_y", "tbl_z",
    "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
    "grip", "hand_conf", "occluded", "handedness",
    "cam_x", "cam_y", "cam_z",
]

IDENTITY_ROT6D = [1, 0, 0, 0, 1, 0]


def make_row(t, tbl=(0, 0, 0), rot6d=IDENTITY_ROT6D, grip=0, hand_conf=1.0, occluded=0,
             handedness="Right", cam=(0, 0, 0)):
    return [t, *tbl, *rot6d, grip, hand_conf, occluded, handedness, *cam]


def make_df(rows):
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


# --- serializer: hand-made deltas ---

def test_format_all_zero_only_gripper():
    assert format_action_text(0, 0, 0, 0, 0, 0, "open", "Right") == "Right arm: open gripper"


def test_format_negative_values():
    text = format_action_text(-5, 0, 3, 0, 0, 0, "close", "Right")
    assert text == "Right arm: move back 5 cm, move up 3 cm, close gripper"


def test_format_order_and_all_components():
    text = format_action_text(5, 2, -3, 10, -10, -5, "close", "Right")
    assert text == (
        "Right arm: move forward 5 cm, move down 3 cm, move left 2 cm, "
        "tilt left 10 degrees, tilt forward 10 degrees, rotate clockwise 5 degrees, close gripper"
    )


def test_format_left_handed_prefix():
    text = format_action_text(0, 0, 0, 0, 0, 0, "open", "Left")
    assert text.startswith("Left arm: ")


# --- rounding edges ---

def test_round_cm_small_delta_omitted():
    assert round_cm(0.0049) == 0  # 0.49 cm -> 0


def test_round_cm_rounds_to_integer():
    assert round_cm(0.052) == 5  # 5.2 cm -> 5


def test_round_deg5_rounding_edges_are_consistent():
    # Banker's rounding on the /5 quotient: deterministic, not necessarily "round half up".
    assert round_deg5(2.5) == round_deg5(2.5)
    assert round_deg5(2.5) in (0, 5)
    assert round_deg5(2.5) == 0
    assert round_deg5(7.5) == 10
    assert round_deg5(-2.5) == 0


def test_round_deg5_rounds_to_nearest_multiple():
    assert round_deg5(12.0) == 10
    assert round_deg5(13.0) == 15


# --- rot6d reconstruction ---

def test_rot6d_round_trip_identity():
    m = rot6d_to_matrix(IDENTITY_ROT6D)
    np.testing.assert_allclose(m, np.eye(3), atol=1e-8)


def test_rot6d_round_trip_arbitrary():
    rot6d = [0, 1, 0, -1, 0, 0]  # 90 degree yaw
    m = rot6d_to_matrix(rot6d)
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-8)
    assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-8)


# --- compute_window_action ---

def test_window_action_translation_and_grip():
    rows = [
        make_row(0.0, tbl=(0.30, 0.0, 0.20), grip=0),
        make_row(0.5, tbl=(0.33, 0.0, 0.17), grip=0),
        make_row(0.99, tbl=(0.35, 0.0, 0.15), grip=1),
    ]
    action = compute_window_action(make_df(rows), window=0, t_start=0.0, t_end=1.0)
    assert action["dx_cm"] == 5
    assert action["dz_cm"] == -5
    assert action["dy_cm"] == 0
    assert action["grip_end"] == "close"
    assert action["occluded_frac"] == 0.0
    assert action["rot_valid"] is True
    assert "move left" not in action["text"] and "move right" not in action["text"]


def test_window_action_fully_occluded_window():
    rows = [
        make_row(0.0, tbl=(0.30, 0.0, 0.20), grip=0, occluded=1, hand_conf=0.1),
        make_row(0.99, tbl=(0.40, 0.0, 0.20), grip=0, occluded=1, hand_conf=0.1),
    ]
    action = compute_window_action(make_df(rows), window=0, t_start=0.0, t_end=1.0)
    assert action["occluded_frac"] == 1.0
    assert action["rot_valid"] is False
    assert action["roll_deg"] == 0 and action["pitch_deg"] == 0 and action["yaw_deg"] == 0
    assert "tilt" not in action["text"] and "rotate" not in action["text"]


def test_window_action_low_confidence_forces_rot_invalid():
    rows = [
        make_row(0.0, tbl=(0.30, 0.0, 0.20), hand_conf=0.1),
        make_row(0.99, tbl=(0.30, 0.0, 0.20), hand_conf=0.1),
    ]
    action = compute_window_action(make_df(rows), window=0, t_start=0.0, t_end=1.0)
    assert action["rot_valid"] is False


# --- integration on synth.py output ---

def test_build_actions_on_synth_track(tmp_path):
    df = generate_synth_track(fps=30)
    track_csv = tmp_path / "track.csv"
    df.to_csv(track_csv, index=False)

    actions = build_actions(track_csv, window_s=1.0, fps=30)

    assert len(actions) == 20
    assert all(a["occluded_frac"] == 0.0 for a in actions)
    # reach phase: gripper stays open
    assert actions[0]["grip_end"] == "open"
    # grasp happens inside window [4,5)
    assert actions[4]["grip_end"] == "close"
    # lift phase (windows 4-8): net upward motion
    assert sum(a["dz_cm"] for a in actions[4:8]) > 0
    # move-left phase (windows 8-12): net leftward motion
    assert sum(a["dy_cm"] for a in actions[8:12]) > 0
    # release happens inside window [16,17)
    assert actions[16]["grip_end"] == "open"
    assert actions[-1]["grip_end"] == "open"
