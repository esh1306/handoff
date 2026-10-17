"""Turn a finished run into a LeRobot dataset for training VLA policies.

    python dataset.py runs/<id> --episodes 50

One recorded video becomes many training episodes: each episode re-simulates the run's
final plan with domain randomisation (object positions, rotations, colours, masses,
friction, camera angle), re-aims the plan at the moved objects, and keeps the episode only
if the robot still succeeds. Every frame records two cameras (front + wrist), joint state,
the commanded action, the task instruction and the current EgoLAP language action.

Output: LeRobot v2.1 layout (parquet + mp4 + meta), in datasets/<run id>/. Newer LeRobot
versions read v3; convert with LeRobot's v2.1 -> v3.0 conversion script.
"""
import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from controller import Sim
from robots import ROBOTS
from scene import build, half_height

CODEBASE_VERSION = "v2.1"


# ---------------- randomisation ----------------

def plan_roles(spec, metrics):
    """Which object each part of the plan is about: the grasped object until the gripper
    closes on it, then the destination (support object or target point)."""
    grasped, close_i = None, None
    for i, s in enumerate(metrics["steps"]):
        if s["action"] == "close" and s["grasped"]:
            grasped, close_i = s["grasped"][0], i
            break
    dest = None
    for c in spec["success"]:
        if c["type"] == "object_on" and c["object"] == grasped:
            dest = ("object", c["support"])
        elif c["type"] == "object_near" and c["object"] == grasped and dest is None:
            dest = ("point", tuple(c["target"]))
    return grasped, close_i, dest


def footprint(o):
    s = o["size"]
    return math.hypot(s[0], s[1]) if o["shape"] in ("box", "container", "mesh") else s[0]


def randomise(spec, roles, rng, pos_jitter, yaw_jitter):
    """A perturbed copy of the spec whose plan follows the moved objects; None if no layout fits."""
    grasped, close_i, dest = roles
    for _ in range(30):
        s = copy.deepcopy(spec)
        offsets, dyaw = {}, {}
        for o in s["objects"]:
            off = np.array([*rng.uniform(-pos_jitter, pos_jitter, 2), 0.0])
            o["pos"] = (np.asarray(o["pos"], float) + off).tolist()
            o["pos"][2] = max(o["pos"][2], half_height(o))
            dyaw[o["name"]] = float(rng.uniform(-yaw_jitter, yaw_jitter))
            o["yaw"] = o.get("yaw", 0) + dyaw[o["name"]]
            o["rgba"] = np.clip(np.asarray(o.get("rgba", [0.6, 0.6, 0.6, 1]), float)
                                + [*rng.uniform(-0.08, 0.08, 3), 0], 0, 1).tolist()
            o["mass"] = o.get("mass", 0.05) * float(rng.uniform(0.7, 1.3))
            o["friction"] = float(rng.uniform(0.9, 1.5))
            offsets[o["name"]] = off
        if _overlaps(spec["objects"], s["objects"]):
            continue  # try another layout

        dest_off = offsets.get(dest[1], np.zeros(3)) if dest and dest[0] == "object" else np.zeros(3)
        for i, step in enumerate(s["plan"]):
            before = close_i is None or i < close_i
            off = offsets.get(grasped, np.zeros(3)) if before else dest_off
            if step["action"] == "move_to":
                step["pos"] = (np.asarray(step["pos"], float) + off).tolist()
                if before and grasped:
                    step["yaw"] = step.get("yaw", 0) + dyaw[grasped]
        for c in s["success"]:
            if c["type"] == "object_near":
                c["target"] = (np.asarray(c["target"], float) + dest_off).tolist()
        return s
    return None


def _overlaps(before, after):
    """True if any two objects got closer than their clearance (or than they started, if nearer)."""
    for i in range(len(after)):
        for j in range(i + 1, len(after)):
            d0 = np.linalg.norm(np.subtract(before[i]["pos"][:2], before[j]["pos"][:2]))
            d1 = np.linalg.norm(np.subtract(after[i]["pos"][:2], after[j]["pos"][:2]))
            need = min(d0, footprint(after[i]) + footprint(after[j]) + 0.01)
            if d1 < need - 1e-6:
                return True
    return False


# ---------------- recording ----------------

class Recorder:
    """Samples the simulation at the dataset fps: images, state, action, language."""

    def __init__(self, sim, plan, fps, video_dir, episode, size):
        self.sim, self.plan, self.fps = sim, plan, fps
        m, robot = sim.m, sim.robot
        self.arm_q = list(sim.qadr)
        self.wrist_q = [m.joint(m.actuator(robot.wrist_actuator).trnid[0]).qposadr[0]] if robot.wrist_actuator else []
        self.finger_q = [m.joint(j).qposadr[0] for j in robot.gripper_joints]
        self.next_t = 0.0
        self.rows = []
        self.writers = {}
        for cam in ("front", "wrist"):
            path = video_dir / f"observation.images.{cam}" / f"episode_{episode:06d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            w = imageio_ffmpeg.write_frames(str(path), (size[1], size[0]), fps=fps, codec="libx264",
                                            quality=8, macro_block_size=8)
            w.send(None)
            self.writers[cam] = w
        self.img_sum = {c: np.zeros(3) for c in self.writers}
        self.img_sq = {c: np.zeros(3) for c in self.writers}
        self.img_min = {c: np.ones(3) for c in self.writers}
        self.img_max = {c: np.zeros(3) for c in self.writers}
        self.img_n = 0

    def gripper_closed_fraction(self):
        r = self.sim.robot
        q = float(np.mean(self.sim.d.qpos[self.finger_q]))
        return float(np.clip((q - r.gripper_qpos_open) / (r.gripper_qpos_closed - r.gripper_qpos_open), 0, 1))

    def __call__(self, sim):
        d = sim.d
        if d.time + 1e-9 < self.next_t:
            return
        self.next_t += 1.0 / self.fps
        front, wrist = sim.render(), sim.render("wrist")
        self.writers["front"].send(np.ascontiguousarray(front))
        self.writers["wrist"].send(np.ascontiguousarray(wrist))
        if len(self.rows) % 5 == 0:  # image stats on a subsample
            for cam, img in (("front", front), ("wrist", wrist)):
                px = img.reshape(-1, 3).astype(np.float64) / 255
                self.img_sum[cam] += px.mean(0)
                self.img_sq[cam] += (px ** 2).mean(0)
                self.img_min[cam] = np.minimum(self.img_min[cam], px.min(0))
                self.img_max[cam] = np.maximum(self.img_max[cam], px.max(0))
            self.img_n += 1
        step = self.plan[max(sim.step_index, 0)] if self.plan else {}
        wrist_cmd = [float(d.ctrl[sim.wrist_act])] if sim.wrist_act is not None else []
        self.rows.append({
            "observation.state": [*d.qpos[self.arm_q], *d.qpos[self.wrist_q], self.gripper_closed_fraction()],
            "observation.ee_pos": d.site_xpos[sim.site].tolist(),
            "action": [*sim.q_cmd, *wrist_cmd, float(sim.grip_closed)],
            "timestamp": (len(self.rows)) / self.fps,
            "step_index": max(sim.step_index, 0),
            "language_action": step.get("language", ""),
            "reason": step.get("reason", ""),
        })

    def close(self):
        for w in self.writers.values():
            w.close()

    def image_stats(self):
        out = {}
        for cam in self.writers:
            n = max(self.img_n, 1)
            mean = self.img_sum[cam] / n
            std = np.sqrt(np.maximum(self.img_sq[cam] / n - mean ** 2, 0))
            out[f"observation.images.{cam}"] = {
                "min": self.img_min[cam].reshape(3, 1, 1).tolist(), "max": self.img_max[cam].reshape(3, 1, 1).tolist(),
                "mean": mean.reshape(3, 1, 1).tolist(), "std": std.reshape(3, 1, 1).tolist(), "count": [len(self.rows)]}
        return out


def vec_stats(rows, key):
    a = np.asarray([r[key] for r in rows], dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
            "std": a.std(0).tolist(), "count": [len(a)]}


# ---------------- export ----------------

def joint_names(robot):
    names = list(robot.arm_joints) + ([robot.wrist_actuator] if robot.wrist_actuator else [])
    return names


def export(run_dir, out_dir, episodes, seed, fps, size, pos_jitter, yaw_jitter, progress=print):
    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    final = json.loads((run / "final.json").read_text(encoding="utf-8"))
    it = run / f"iter_{final['iteration']}"
    spec = json.loads((it / "spec.json").read_text(encoding="utf-8"))
    metrics = json.loads((it / "metrics.json").read_text(encoding="utf-8"))
    robot = ROBOTS[cfg["robot"]]
    task = spec.get("task_description") or cfg.get("task") or "do the task"
    roles = plan_roles(spec, metrics)
    if roles[0] is None:
        raise RuntimeError("The run's final round never grasped anything, so there is no task to randomise")

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    names = joint_names(robot)
    state_names = names + ["gripper_closed"]
    action_names = [f"{n}.target" for n in names] + ["gripper_close"]

    kept, tried, total_frames, ep_meta, ep_stats = 0, 0, 0, [], []
    t0 = time.time()
    while kept < episodes and tried < episodes * 3:
        tried += 1
        s = spec if tried == 1 else randomise(spec, roles, rng, pos_jitter, yaw_jitter)
        if s is None:
            continue
        model, _ = build(s, robot)
        jitter = (0, 0) if tried == 1 else (float(rng.uniform(-8, 8)), float(rng.uniform(-4, 4)))
        sim = Sim(model, robot, s, camera_jitter=jitter)
        sim.renderer = mujoco.Renderer(model, *size)
        tmp_videos = out / "_tmp"
        rec = Recorder(sim, s["plan"], fps, tmp_videos, kept, size)
        sim.recorder = rec
        m = sim.run(s["plan"])
        rec.close()
        sim.close()
        if not m["all_success"]:
            shutil.rmtree(tmp_videos, ignore_errors=True)
            progress(f"episode attempt {tried}: robot failed, discarded ({kept}/{episodes} kept)", kept, tried)
            continue

        # keep: move videos into place and write the parquet
        for cam_dir in tmp_videos.iterdir():
            dst = out / "videos" / "chunk-000" / cam_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in cam_dir.iterdir():
                shutil.move(str(f), dst / f.name)
        shutil.rmtree(tmp_videos, ignore_errors=True)
        n = len(rec.rows)
        table = pa.table({
            "observation.state": pa.array([r["observation.state"] for r in rec.rows], pa.list_(pa.float32())),
            "observation.ee_pos": pa.array([r["observation.ee_pos"] for r in rec.rows], pa.list_(pa.float32())),
            "action": pa.array([r["action"] for r in rec.rows], pa.list_(pa.float32())),
            "timestamp": pa.array([r["timestamp"] for r in rec.rows], pa.float32()),
            "frame_index": pa.array(range(n), pa.int64()),
            "episode_index": pa.array([kept] * n, pa.int64()),
            "index": pa.array(range(total_frames, total_frames + n), pa.int64()),
            "task_index": pa.array([0] * n, pa.int64()),
            "step_index": pa.array([r["step_index"] for r in rec.rows], pa.int64()),
            "language_action": pa.array([r["language_action"] for r in rec.rows], pa.string()),
            "reason": pa.array([r["reason"] for r in rec.rows], pa.string()),
        })
        data_path = out / "data" / "chunk-000" / f"episode_{kept:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)

        stats = {k: vec_stats(rec.rows, k) for k in ("observation.state", "observation.ee_pos", "action", "timestamp")}
        for k, vals in (("frame_index", range(n)), ("episode_index", [kept] * n),
                        ("index", range(total_frames, total_frames + n)), ("task_index", [0] * n),
                        ("step_index", [r["step_index"] for r in rec.rows])):
            stats[k] = vec_stats([{"v": v} for v in vals], "v")
        stats.update(rec.image_stats())
        ep_stats.append({"episode_index": kept, "stats": stats})
        ep_meta.append({"episode_index": kept, "tasks": [task], "length": n,
                        "randomised": tried != 1,
                        "objects": {o["name"]: [round(v, 4) for v in o["pos"]] for o in s["objects"]}})
        total_frames += n
        kept += 1
        progress(f"episode {kept}/{episodes} kept ({n} frames, attempt {tried})", kept, tried)

    if kept == 0:
        raise RuntimeError("No episode succeeded; the plan is too fragile to randomise. Try --pos-jitter 0.01")

    video_info = lambda: {"dtype": "video", "shape": [size[0], size[1], 3], "names": ["height", "width", "channels"],
                          "info": {"video.height": size[0], "video.width": size[1], "video.codec": "h264",
                                   "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": fps,
                                   "video.channels": 3, "has_audio": False}}
    scalar = lambda dtype: {"dtype": dtype, "shape": [1], "names": None}
    info = {
        "codebase_version": CODEBASE_VERSION, "robot_type": robot.name,
        "total_episodes": kept, "total_frames": total_frames, "total_tasks": 1,
        "total_videos": kept * 2, "total_chunks": 1, "chunks_size": 1000, "fps": fps,
        "splits": {"train": f"0:{kept}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.front": video_info(),
            "observation.images.wrist": video_info(),
            "observation.state": {"dtype": "float32", "shape": [len(state_names)], "names": state_names},
            "observation.ee_pos": {"dtype": "float32", "shape": [3], "names": ["x", "y", "z"]},
            "action": {"dtype": "float32", "shape": [len(action_names)], "names": action_names},
            "timestamp": scalar("float32"), "frame_index": scalar("int64"), "episode_index": scalar("int64"),
            "index": scalar("int64"), "task_index": scalar("int64"), "step_index": scalar("int64"),
            "language_action": scalar("string"), "reason": scalar("string"),
        },
    }
    meta = out / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n", encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps({k: e[k] for k in ("episode_index", "tasks", "length")}) + "\n" for e in ep_meta),
        encoding="utf-8")
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ep_stats), encoding="utf-8")
    summary = {"source_run": run.name, "model": cfg.get("model"), "robot": robot.name, "task": task,
               "episodes": kept, "attempts": tried, "success_rate": round(kept / tried, 3),
               "frames": total_frames, "fps": fps, "image_size": list(size), "seed": seed,
               "pos_jitter_m": pos_jitter, "yaw_jitter_deg": yaw_jitter,
               "seconds": round(time.time() - t0, 1), "episodes_detail": ep_meta}
    (out / "generation.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "README.md").write_text(dataset_card(summary, state_names, action_names), encoding="utf-8")
    return summary


def dataset_card(s, state_names, action_names):
    return f"""# {s['task']}

Simulated {s['robot']} demonstrations in MuJoCo, generated by monkeyseemonkeydo from run
`{s['source_run']}` ({s['model']} turned a video of a person into the scene and plan).

- {s['episodes']} episodes, {s['frames']} frames at {s['fps']} fps, {s['image_size'][0]}x{s['image_size'][1]} images
- Domain randomisation: object positions +/-{s['pos_jitter_m'] * 100:.0f} cm, rotation +/-{s['yaw_jitter_deg']:.0f} deg,
  colour, mass, friction, camera angle; episodes where the robot failed were discarded
  ({s['episodes']} kept of {s['attempts']} attempts, {s['success_rate']:.0%})
- Format: LeRobot {CODEBASE_VERSION}

| Feature | Contents |
|---|---|
| `observation.images.front`, `observation.images.wrist` | RGB video |
| `observation.state` | {', '.join(state_names)} (radians; gripper 0 open .. 1 closed) |
| `observation.ee_pos` | tool point position (m, robot base frame) |
| `action` | {', '.join(action_names)} (absolute joint targets; gripper 0/1) |
| `language_action` | EgoLAP-style language action of the current plan step |
| `reason` | the model's one-line reason for that step |
| `task_index` | the instruction in `meta/tasks.jsonl` |

## Using it with LeRobot

Current LeRobot (0.6+) reads v3.0. Convert once, locally (this moves the v2.1 copy to
`<folder>_old`):

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=local/<name> --root=<path to this folder> --push-to-hub=false
```

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/<name>", root="<path to this folder>", video_backend="pyav")
```

`video_backend="pyav"` avoids needing system FFmpeg libraries for torchcodec on Windows.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", help="default: datasets/<run id>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--size", type=int, default=256, help="square image size in pixels")
    ap.add_argument("--pos-jitter", type=float, default=0.04, help="max object shift in metres")
    ap.add_argument("--yaw-jitter", type=float, default=20, help="max object rotation in degrees")
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = Path(args.out) if args.out else Path(__file__).parent / "datasets" / run.name
    progress_file = out.parent / f"{out.name}.progress.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def progress(msg, kept, tried):
        print(msg, flush=True)
        progress_file.write_text(json.dumps({"status": "running", "message": msg, "kept": kept,
                                             "tried": tried, "target": args.episodes}), encoding="utf-8")

    try:
        s = export(run, out, args.episodes, args.seed, args.fps, (args.size, args.size),
                   args.pos_jitter, args.yaw_jitter, progress)
    except Exception as e:
        progress_file.write_text(json.dumps({"status": "failed", "message": f"{type(e).__name__}: {e}"}), encoding="utf-8")
        raise
    progress_file.write_text(json.dumps({"status": "done", "message": f"{s['episodes']} episodes, {s['frames']} frames",
                                         **{k: s[k] for k in ("episodes", "attempts", "success_rate", "frames", "seconds")}}),
                             encoding="utf-8")
    print(f"\n{s['episodes']} episodes ({s['success_rate']:.0%} of attempts succeeded), {s['frames']} frames "
          f"in {s['seconds']} s -> {out}")


if __name__ == "__main__":
    main()
