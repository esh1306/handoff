"""Replay a run's final (or chosen) iteration in the interactive MuJoCo viewer.

    python replay.py runs/<timestamp>            # final iteration
    python replay.py runs/<timestamp> --iter 2   # a specific iteration
"""
import argparse
import json
import time
from pathlib import Path

import mujoco.viewer

from controller import Sim
from robots import ROBOTS
from scene import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--iter", type=int)
    ap.add_argument("--loop", action="store_true", help="replay over and over")
    args = ap.parse_args()

    run = Path(args.run_dir)
    final = json.loads((run / "final.json").read_text(encoding="utf-8"))
    it = args.iter or final["iteration"]
    spec = json.loads((run / f"iter_{it}" / "spec.json").read_text(encoding="utf-8"))
    robot = ROBOTS[final["robot"]]
    model, _ = build(spec, robot)
    print(f"{spec.get('task_description', '')}\nRobot {robot.name}, iteration {it}")

    sim = Sim(model, robot, spec)
    with mujoco.viewer.launch_passive(model, sim.d) as v:
        def tick():  # show the sim at real-time speed
            if not v.is_running():
                raise SystemExit
            v.sync()
            time.sleep(Sim.CTRL_DT)
        sim.on_tick = tick
        while v.is_running():
            for step in spec["plan"]:
                print(f"  {step['action']:8s} {step.get('language', '')}")
            metrics = sim.run(spec["plan"])
            print("  checks:", [(c["type"], c["passed"]) for c in metrics["success"]])
            if not args.loop:
                while v.is_running():
                    v.sync()
                    time.sleep(0.05)
                break
            sim.reset()


if __name__ == "__main__":
    main()
