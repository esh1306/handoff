"""track.csv + actions.jsonl -> replay.mp4: the Franka executes the language actions in MuJoCo.

Physics, not a kinematic replay: each 1 s language action ("move down 5 cm, close
gripper") becomes a robot move, the gripper really has to hold the object, and the
result is measured. If the labels describe the task correctly, the object ends up
where the person put it.

    python -m handoff.replay takes/synth
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from handoff.actions import build_actions
from handoff.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "monkeyseemonkeydo"))
from controller import Sim  # noqa: E402
from robots import ROBOTS  # noqa: E402
from scene import build  # noqa: E402

OBJECT_RADIUS = 0.03       # m; a 4 cm-radius mug is wider than the Panda's 8 cm opening
OBJECT_HALF_HEIGHT = 0.045
COMFORT_REACH = 0.70       # m; beyond this the Panda's grip gets shaky while carrying
WORKSPACE_CENTRE = np.array([0.5, 0.0])  # where an out-of-reach path gets re-centred


def choose_base(track, base):
    """Use the configured base unless the path leaves the comfortable reach; then re-centre it."""
    xy = track[["tbl_x", "tbl_y"]].to_numpy()
    if np.linalg.norm(xy - base[:2], axis=1).max() <= COMFORT_REACH:
        return base, False
    centre = (xy.min(axis=0) + xy.max(axis=0)) / 2
    return np.array([*(centre - WORKSPACE_CENTRE), base[2]]), True


def to_robot(p_tbl, base, max_reach):
    """Table frame -> robot base frame, clamped to the robot's reach."""
    p = np.asarray(p_tbl, float) - base
    r = np.linalg.norm(p[:2])
    if r > max_reach:
        p[:2] *= max_reach / r
    p[2] = max(p[2], 0.01)
    return p


def build_episode(track, actions, base, max_reach):
    """Scene objects, robot plan and success checks for one take."""
    closes = track.index[(track["grip"] == 1) & (track["grip"].shift(fill_value=0) == 0)]
    opens = track.index[(track["grip"] == 0) & (track["grip"].shift(fill_value=0) == 1)]
    first = track.iloc[0]
    start = to_robot(first[["tbl_x", "tbl_y", "tbl_z"]], base, max_reach)

    objects, success = [], []
    if len(closes):
        pick = to_robot(track.loc[closes[0], ["tbl_x", "tbl_y", "tbl_z"]], base, max_reach)
        objects.append({"name": "mug", "shape": "cylinder", "size": [OBJECT_RADIUS, OBJECT_HALF_HEIGHT],
                        "pos": [pick[0], pick[1], OBJECT_HALF_HEIGHT], "rgba": [0.85, 0.2, 0.15, 1],
                        "mass": 0.08})
        success.append({"type": "object_lifted", "object": "mug", "min_height": 0.05})
        if len(opens):
            place = to_robot(track.loc[opens[-1], ["tbl_x", "tbl_y", "tbl_z"]], base, max_reach)
            objects.append({"name": "target", "shape": "cylinder", "size": [0.06, 0.002],
                            "pos": [place[0], place[1], 0.002], "rgba": [0.2, 0.6, 0.3, 0.6], "fixed": True})
            success.append({"type": "object_near", "object": "mug",
                            "target": [place[0], place[1], OBJECT_HALF_HEIGHT], "tolerance": 0.05})

    plan = [{"action": "move_to", "pos": start.round(4).tolist(), "yaw": 0,
             "language": "go to the hand's starting position", "reason": "start of the take"},
            {"action": "open", "language": "open gripper", "reason": "hand starts open"}]
    gripper = "open"
    for a in actions:
        delta = [a["dx_cm"], a["dy_cm"], a["dz_cm"]]
        if any(delta):
            plan.append({"action": "move_by", "delta_cm": delta, "window": a["window"],
                         "language": a["text"], "reason": f"window {a['window']} ({a['t_start']:.0f}-{a['t_end']:.0f} s)"})
        if a["grip_end"] != gripper:
            gripper = a["grip_end"]
            plan.append({"action": "close" if gripper == "close" else "open", "window": a["window"],
                         "language": f"{gripper} gripper", "reason": f"hand {gripper}s in window {a['window']}"})
    camera = {"lookat": [0.45, 0.25, 0.05], "distance": 1.5, "azimuth": 200, "elevation": -28}
    return {"task": "replay of the take's language actions", "camera": camera,
            "objects": objects, "plan": plan, "success": success}


def replay(take_dir, cfg=None):
    take = Path(take_dir)
    cfg = cfg or load_config()
    track = pd.read_csv(take / "track.csv")
    actions_path = take / "actions.jsonl"
    if actions_path.exists():
        actions = [json.loads(line) for line in actions_path.read_text().splitlines() if line.strip()]
    else:
        actions = build_actions(take / "track.csv", window_s=cfg.window_s, fps=cfg.fps)

    base, recentred = choose_base(track, np.asarray(cfg.replay.robot_base_in_table, float))
    if recentred:
        print(f"path leaves the Panda's comfortable reach from the configured base; "
              f"using base {base.round(3).tolist()} in the table frame instead")
    episode = build_episode(track, actions, base, cfg.replay.max_reach_m)
    episode["robot_base_in_table"] = base.round(4).tolist()
    robot = ROBOTS["panda"]
    model, _ = build(episode, robot)
    sim = Sim(model, robot, episode, video_path=take / "replay.mp4")
    metrics = sim.run(episode["plan"])
    sim.close()

    for step, info in zip(episode["plan"], metrics["steps"]):
        info["window"] = step.get("window")
        if info.get("tcp_error_mm", 0) > 20:
            info["unreachable"] = True
    (take / "replay_episode.json").write_text(json.dumps(episode, indent=1))
    (take / "replay.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take", help="take directory containing track.csv")
    args = parser.parse_args()
    m = replay(args.take)
    for c in m["success"]:
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['type']}({c['object']}): {c['detail']}")
    bad = [s for s in m["steps"] if s.get("unreachable")]
    print(f"{len(m['steps'])} steps, {len(bad)} missed by more than 2 cm; wrote {Path(args.take) / 'replay.mp4'}")


if __name__ == "__main__":
    main()
