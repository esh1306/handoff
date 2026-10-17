# video2sim

Video of a person doing a tabletop task in, MuJoCo simulation of a robot doing the same
task out. A vision model (GPT-6 Astra by default) watches the video and writes the scene
plus a robot plan; MuJoCo runs it with real physics and measures the result; the model
compares the simulation with the video and fixes the plan, until the task succeeds.

This is the "real2sim" loop from the Gemini 4 Argon post, with EgoLAP-style language
actions ("move down 12 cm, close gripper") and a one-line reason per step.

## Setup

```bash
pip install -r video2sim/requirements.txt
# Franka Panda model (either location works):
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
echo OPENAI_API_KEY=sk-... > video2sim/.env     # never commit this file
```

## Use

```bash
cd video2sim
python server.py                 # web UI at http://127.0.0.1:8765
python loop.py --video clip.mp4 --robot panda --model gpt-6-astra   # same thing from the CLI
python replay.py runs/<id>       # interactive 3D viewer of a run
python make_sample.py            # re-render the sample clip
```

Models: `gpt-6-astra`, `gpt-6.1-sol`, `gemini-4-argon`, `claude-opus-5-5`, or `mock`
(no API call, for testing). Robots: `panda` (Franka, grips from above) or `qarm`
(4-DOF QArm lookalike, grips from the side).

## Files

| File | What it does |
|---|---|
| `loop.py` | The analyse, simulate, critique loop; writes `runs/<id>/` |
| `server.py`, `static/index.html` | Web UI: start runs, live progress, results |
| `providers.py` | Swappable model backends (OpenAI, Gemini, Anthropic, mock) |
| `prompts.py` | What the model is asked |
| `scene.py` | JSON scene spec to MuJoCo model |
| `controller.py` | IK, plan execution, video recording, success checks |
| `robots.py` | Panda and QArm adapters |
| `replay.py` | Interactive viewer |
