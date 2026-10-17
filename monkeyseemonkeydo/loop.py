"""Video -> MuJoCo simulation of a robot doing the same task.

    python loop.py --video demo.mp4 --robot panda --model gemini-4-argon

Loop: the model watches the video and writes a scene + robot plan (JSON); we build the
scene in MuJoCo, run the plan, render it and measure what happened; the model compares
the simulation with the video and revises. Stops when every success check passes and the
model says it matches, or after --iters attempts.

Each run is saved under runs/<timestamp>/ (spec, metrics, simulation video per iteration,
and events.jsonl, a live log the web server streams). Watch the result with:
    python replay.py runs/<timestamp>
"""
import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import cv2

from controller import Sim
from prompts import analyse_prompt, critique_prompt
from providers import get_provider
from robots import ROBOTS
from scene import build, validate
from video import extract_frames, pick, to_jpeg


class Events:
    """Append-only JSON-lines log of what the run is doing."""

    def __init__(self, path):
        self.path = path

    def __call__(self, type, **data):
        event = {"t": round(time.time(), 2), "type": type, **data}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        msg = data.get("message") or ""
        print(f"[{type}] {msg}", flush=True)


def parse_json(text):
    """Pull the first JSON object out of a model reply."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


def ask(provider, prompt, images, video_path, log_path, emit, stage, retries=2):
    """Call the model until it returns a spec that parses and validates."""
    for attempt in range(retries + 1):
        emit("model_call", stage=stage, attempt=attempt + 1, images=len(images),
             message=f"asking {provider.model} ({stage}, attempt {attempt + 1})")
        t = time.time()
        reply = provider.generate(prompt, images, video_path)
        log_path.write_text(reply, encoding="utf-8")
        try:
            spec = parse_json(reply)
            errs = validate(spec)
        except (json.JSONDecodeError, ValueError) as e:
            errs = [f"reply is not valid JSON: {e}"]
        emit("model_reply", stage=stage, attempt=attempt + 1, seconds=round(time.time() - t, 1),
             valid=not errs, problems=errs, reply_file=log_path.name,
             message=f"reply in {time.time() - t:.0f}s" + (f", invalid: {'; '.join(errs)[:200]}" if errs else ""))
        if not errs:
            return spec
        prompt += ("\n\nYour last reply had these problems, fix them and reply with only the JSON:\n- "
                   + "\n- ".join(errs))
    raise RuntimeError("The model did not return a valid scene after retries")


def run(args, run_dir, emit):
    robot = ROBOTS[args.robot]
    provider = get_provider(args.model, args.provider, args.robot)
    emit("started", robot=robot.name, model=args.model, iters=args.iters,
         message=f"robot {robot.name}, model {args.model}")

    info, frames = extract_frames(args.video, args.frames)
    video_imgs = [(label, to_jpeg(rgb)) for label, rgb in frames]
    for i, (_, rgb) in enumerate(frames):
        cv2.imwrite(str(run_dir / f"video_{i}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (run_dir / "video_frames.json").write_text(json.dumps([l for l, _ in frames]), encoding="utf-8")
    emit("frames", count=len(frames), info=info, message=f"extracted {len(frames)} frames ({info['duration_s']} s video)")
    video_arg = args.video if provider.native_video else None

    # The closing axis depends only on the robot; build an empty scene to read it.
    empty = {"objects": [], "plan": [], "success": []}
    axis = Sim(build(empty, robot)[0], robot, empty).closing_axis()

    spec = ask(provider, analyse_prompt(robot.notes, axis, info, args.task),
               video_imgs, video_arg, run_dir / "reply_0.txt", emit, "analyse")
    best, it = None, 0
    for it in range(1, args.iters + 1):
        it_dir = run_dir / f"iter_{it}"
        it_dir.mkdir()
        (it_dir / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        emit("iteration", iteration=it, task=spec.get("task_description", ""),
             message=f"iteration {it}: simulating {len(spec['plan'])} steps")
        sim = None
        try:
            model, _ = build(spec, robot)
            sim = Sim(model, robot, spec, video_path=it_dir / "sim.mp4")
            metrics = sim.run(spec["plan"])
        except Exception as e:  # bad geometry etc. - report it back to the model
            metrics = {"error": f"{type(e).__name__}: {e}", "all_success": False}
            if sim:
                sim.close()
            sim = None
        (it_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
        sim_imgs = pick(sim.keyframes, max(args.frames, 4)) if sim else []
        if sim:
            sim.close()
        for i, (label, rgb) in enumerate(sim_imgs):
            cv2.imwrite(str(it_dir / f"sim_{i}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        (it_dir / "sim_frames.json").write_text(json.dumps([l for l, _ in sim_imgs]), encoding="utf-8")
        checks = metrics.get("success", [])
        emit("simulated", iteration=it, all_success=metrics["all_success"], model_done=bool(spec.get("done")),
             error=metrics.get("error"),
             message=", ".join(f"{c['type']}({c['object']}) {'pass' if c['passed'] else 'fail'}" for c in checks)
             or metrics.get("error", "no checks"))
        if metrics["all_success"]:
            best = it
        if metrics["all_success"] and spec.get("done"):
            break
        if it == args.iters:
            break
        images = video_imgs + [(label, to_jpeg(rgb)) for label, rgb in sim_imgs]
        spec = ask(provider, critique_prompt(spec, metrics, it + 1, args.iters),
                   images, video_arg, it_dir / "reply.txt", emit, f"critique {it}")

    final = best or it
    (run_dir / "final.json").write_text(json.dumps(
        {"iteration": final, "robot": robot.name, "model": args.model,
         "all_success": best is not None}, indent=1), encoding="utf-8")
    emit("done", iteration=final, all_success=best is not None,
         message=f"final iteration {final}: " + ("all checks pass" if best else "checks still failing"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--robot", choices=sorted(ROBOTS), default="panda")
    ap.add_argument("--model", default="gpt-6-astra",
                    help="e.g. gpt-6-astra, gpt-6.1-sol, gemini-4-argon, claude-opus-5-5, or mock")
    ap.add_argument("--provider", default="auto", choices=["auto", "gemini", "anthropic", "openai", "mock"])
    ap.add_argument("--iters", type=int, default=5, help="max simulate/critique rounds")
    ap.add_argument("--frames", type=int, default=8, help="video frames sent to the model")
    ap.add_argument("--task", help="optional hint describing the task")
    ap.add_argument("--out", default=str(Path(__file__).parent / "runs"))
    ap.add_argument("--run-dir", help="exact folder for this run (default: <out>/<timestamp>)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ("out", "run_dir")}
    (run_dir / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    emit = Events(run_dir / "events.jsonl")
    print(f"Run folder: {run_dir}")
    try:
        run(args, run_dir, emit)
    except Exception as e:
        emit("error", message=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        return 1
    print(f"Watch it live: python replay.py {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
