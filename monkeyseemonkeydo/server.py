"""Web server + UI for the video -> simulation pipeline.

    python server.py            # then open http://127.0.0.1:8765  (--port to change)

Each run is a separate `loop.py` process writing into runs/<id>/; the server starts and
stops those processes, streams their events.jsonl to the browser and serves every file
they produce. API keys come from the environment or monkeyseemonkeydo/.env. Binds to localhost only.
"""
import asyncio
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from providers import KEY_ENV, resolve_provider
from robots import ROBOTS

HERE = Path(__file__).parent
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)
PY = sys.executable
SAMPLES = HERE / "samples"
DATASETS = HERE / "datasets"
MODELS = [  # (id, what it's good for); the first available one is the default
    ("gpt-6-astra", "OpenAI's most capable model, best at spatial reasoning (slower, priciest)"),
    ("gpt-6.1-sol", "OpenAI's balanced model, faster and about 5x cheaper"),
    ("gemini-4-argon", "Google's robotics model from the blog post"),
    ("claude-opus-5-5", "Anthropic's most capable everyday model"),
    ("mock", "Canned answer, no API call; checks the pipeline in seconds"),
]

app = FastAPI(title="monkeyseemonkeydo")
procs: dict[str, subprocess.Popen] = {}


def to_web_mp4(src, dst):
    """Phone clips (HEVC .mov, rotation metadata) -> upright 720p H.264 any browser and OpenCV can read."""
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-vf", "scale=-2:'min(720,ih)'", "-movflags", "+faststart", str(dst)],
                   check=True, capture_output=True)


def run_path(run_id):
    p = (RUNS / run_id).resolve()
    if p.parent != RUNS.resolve() or not p.is_dir():
        raise HTTPException(404, f"No run {run_id!r}")
    return p


def read_json(p, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_events(p):
    try:
        lines = (p / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # line still being written
    return out


def status_of(run_id, events):
    proc = procs.get(run_id)
    if (RUNS / run_id / "stopped").exists():
        return "stopped"
    if events and events[-1]["type"] == "done":
        return "done"
    if events and events[-1]["type"] == "error":
        return "failed"
    if proc and proc.poll() is None:
        return "running"
    return "interrupted"


def summary(p):
    events = read_events(p)
    cfg = read_json(p / "config.json", {})
    final = read_json(p / "final.json")
    task = next((e.get("task") for e in reversed(events) if e.get("task")), "")
    return {"id": p.name, "status": status_of(p.name, events), "robot": cfg.get("robot"),
            "model": cfg.get("model"), "task": task, "created": p.stat().st_ctime,
            "iterations": sum(1 for e in events if e["type"] == "simulated"),
            "all_success": bool(final and final.get("all_success"))}


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


def has_key(prov):
    if prov == "mock":
        return True
    return bool(os.environ.get(KEY_ENV[prov]) or (prov == "gemini" and os.environ.get("GOOGLE_API_KEY")))


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    """Presenter view: /demo#<run id>"""
    return (HERE / "static" / "demo.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config():
    models = [{"id": m, "about": about, "available": has_key(resolve_provider(m))} for m, about in MODELS]
    samples = [{"id": p.stem, "label": p.stem.replace("_", " ").capitalize(), "src": f"/samples/{p.name}"}
               for p in sorted(SAMPLES.glob("*.mp4"))]
    return {"robots": {k: r.notes for k, r in ROBOTS.items()}, "models": models, "samples": samples}


@app.get("/api/runs")
def list_runs():
    runs = [summary(p) for p in RUNS.iterdir() if p.is_dir()]
    return sorted(runs, key=lambda r: r["created"], reverse=True)


@app.post("/api/runs")
async def start_run(video: UploadFile | None = File(None), sample: str = Form(""),
                    robot: str = Form("panda"), model: str = Form(MODELS[0][0]),
                    provider: str = Form("auto"), iters: int = Form(5), frames: int = Form(8),
                    task: str = Form("")):
    if robot not in ROBOTS:
        raise HTTPException(400, f"Unknown robot {robot!r}")
    try:
        prov = resolve_provider(model.strip(), provider)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not has_key(prov):
        raise HTTPException(400, f"No API key for {prov}: add {KEY_ENV[prov]}=... to monkeyseemonkeydo/.env and restart the server")
    sample_file = SAMPLES / f"{Path(sample).name}.mp4" if sample else None
    if sample_file and not sample_file.exists():
        raise HTTPException(404, f"No sample {sample!r}")
    if not video and not sample_file:
        raise HTTPException(400, "Upload a video or pick a sample")

    run_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    run_dir = RUNS / run_id
    run_dir.mkdir()
    video_file = run_dir / "input.mp4"
    if sample_file:
        shutil.copyfile(sample_file, video_file)
    else:
        ext = Path(video.filename or "video.mp4").suffix.lower() or ".mp4"
        original = run_dir / f"original{ext}"
        with open(original, "wb") as f:
            shutil.copyfileobj(video.file, f)
        try:
            await asyncio.to_thread(to_web_mp4, original, video_file)
        except subprocess.CalledProcessError as e:
            shutil.rmtree(run_dir)
            raise HTTPException(400, f"Couldn't read that video: {e.stderr.decode(errors='replace')[-300:]}")
    env = os.environ.copy()

    cmd = [PY, str(HERE / "loop.py"), "--video", str(video_file), "--robot", robot,
           "--model", model.strip(), "--provider", provider, "--iters", str(max(1, min(iters, 20))),
           "--frames", str(max(2, min(frames, 24))), "--run-dir", str(run_dir)]
    if task.strip():
        cmd += ["--task", task.strip()]
    log = open(run_dir / "process.log", "w", encoding="utf-8")
    procs[run_id] = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"id": run_id}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    p = run_path(run_id)
    events = read_events(p)
    iterations = []
    for i in range(1, 100):
        d = p / f"iter_{i}"
        if not d.is_dir():
            break
        reply = d / "reply.txt"
        iterations.append({
            "n": i,
            "spec": read_json(d / "spec.json"),
            "metrics": read_json(d / "metrics.json"),
            "video": f"/runs/{run_id}/iter_{i}/sim.mp4" if (d / "sim.mp4").exists() else None,
            "frames": [{"label": l, "src": f"/runs/{run_id}/iter_{i}/sim_{k}.jpg"}
                       for k, l in enumerate(read_json(d / "sim_frames.json", []))],
            "critique": reply.read_text(encoding="utf-8") if reply.exists() else None,
        })
    first = p / "reply_0.txt"
    inputs = sorted(p.glob("input.*"))
    log = p / "process.log"
    return {**summary(p), "config": read_json(p / "config.json", {}), "events": events,
            "final": read_json(p / "final.json"),
            "input_video": f"/runs/{run_id}/{inputs[0].name}" if inputs else None,
            "video_frames": [{"label": l, "src": f"/runs/{run_id}/video_{k}.jpg"}
                             for k, l in enumerate(read_json(p / "video_frames.json", []))],
            "analysis": first.read_text(encoding="utf-8") if first.exists() else None,
            "iterations": iterations,
            "process_log": log.read_text(encoding="utf-8", errors="replace")[-20000:] if log.exists() else ""}


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str, start: int = 0):
    p = run_path(run_id)

    async def gen():
        sent = start
        while True:
            events = read_events(p)
            for e in events[sent:]:
                yield f"data: {json.dumps(e)}\n\n"
            sent = len(events)
            status = status_of(run_id, events)
            if status != "running":
                yield f"event: end\ndata: {json.dumps({'status': status})}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    p = run_path(run_id)
    proc = procs.get(run_id)
    if proc and proc.poll() is None:
        proc.terminate()
        (p / "stopped").touch()
    return summary(p)


@app.post("/api/runs/{run_id}/replay")
def replay(run_id: str, iteration: int | None = None):
    """Open the interactive MuJoCo viewer on this machine."""
    p = run_path(run_id)
    if not (p / "final.json").exists() and iteration is None:
        raise HTTPException(400, "This run has no final iteration yet; pick one")
    if iteration is not None and not (p / f"iter_{iteration}" / "spec.json").exists():
        raise HTTPException(404, f"No iteration {iteration}")
    if not (p / "final.json").exists():  # replay.py reads the robot from final.json
        cfg = read_json(p / "config.json", {})
        (p / "final.json").write_text(json.dumps({"iteration": iteration, "robot": cfg.get("robot", "panda"),
                                                  "model": cfg.get("model"), "all_success": False}))
    cmd = [PY, str(HERE / "replay.py"), str(p), "--loop"] + (["--iter", str(iteration)] if iteration else [])
    subprocess.Popen(cmd, cwd=HERE)
    return {"ok": True}


@app.post("/api/runs/{run_id}/dataset")
def make_dataset(run_id: str, episodes: int = 50):
    """Generate a LeRobot training dataset from the run (domain-randomised re-simulations)."""
    p = run_path(run_id)
    if not (p / "final.json").exists():
        raise HTTPException(400, "Finish the run first")
    key = f"dataset:{run_id}"
    if key in procs and procs[key].poll() is None:
        raise HTTPException(409, "Already making a dataset for this run")
    DATASETS.mkdir(exist_ok=True)
    (DATASETS / f"{run_id}.progress.json").write_text(json.dumps(
        {"status": "running", "message": "starting", "kept": 0, "tried": 0, "target": episodes}))
    log = open(DATASETS / f"{run_id}.log", "w", encoding="utf-8")
    procs[key] = subprocess.Popen([PY, str(HERE / "dataset.py"), str(p), "--episodes", str(max(1, min(episodes, 1000)))],
                                  cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    return {"ok": True}


@app.get("/api/runs/{run_id}/dataset")
def dataset_status(run_id: str):
    run_path(run_id)
    prog = read_json(DATASETS / f"{run_id}.progress.json")
    if prog is None:
        return {"status": "none"}
    proc = procs.get(f"dataset:{run_id}")
    if prog.get("status") == "running" and (proc is None or proc.poll() is not None):
        log = DATASETS / f"{run_id}.log"
        tail = log.read_text(encoding="utf-8", errors="replace")[-600:] if log.exists() else ""
        prog = {"status": "failed", "message": tail.strip().splitlines()[-1] if tail.strip() else "stopped"}
    out = {**prog}
    gen = read_json(DATASETS / run_id / "generation.json")
    if prog.get("status") == "done" and gen:
        n = min(gen["episodes"], 4)
        out["summary"] = {k: v for k, v in gen.items() if k != "episodes_detail"}
        out["preview"] = [{"episode": e, "front": f"/datasets/{run_id}/videos/chunk-000/observation.images.front/episode_{e:06d}.mp4",
                           "wrist": f"/datasets/{run_id}/videos/chunk-000/observation.images.wrist/episode_{e:06d}.mp4"}
                          for e in range(n)]
        out["features"] = read_json(DATASETS / run_id / "meta" / "info.json", {}).get("features")
    return out


@app.get("/api/runs/{run_id}/dataset/zip")
def dataset_zip(run_id: str):
    run_path(run_id)
    d = DATASETS / run_id
    if not (d / "meta" / "info.json").exists():
        raise HTTPException(404, "No dataset for this run yet")
    archive = shutil.make_archive(str(Path(tempfile.mkdtemp()) / f"{run_id}-lerobot"), "zip", d)
    return FileResponse(archive, filename=f"{run_id}-lerobot.zip")


@app.get("/api/runs/{run_id}/zip")
def download(run_id: str):
    p = run_path(run_id)
    out = Path(tempfile.mkdtemp()) / run_id
    archive = shutil.make_archive(str(out), "zip", p)
    return FileResponse(archive, filename=f"{run_id}.zip")


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    p = run_path(run_id)
    proc = procs.pop(run_id, None)
    if proc and proc.poll() is None:
        proc.terminate()
        proc.wait(5)
    shutil.rmtree(p)
    return {"ok": True}


app.mount("/runs", StaticFiles(directory=RUNS), name="runs")
SAMPLES.mkdir(exist_ok=True)
app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
DATASETS.mkdir(exist_ok=True)
app.mount("/datasets", StaticFiles(directory=DATASETS), name="datasets")

if __name__ == "__main__":
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    port = ap.parse_args().port
    print(f"Open http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
