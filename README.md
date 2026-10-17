# Handoff

Turn an ordinary phone video of a person doing a tabletop task into robot training data.

The repo has two pipelines that share the same language-action format (EgoLAP: "move down
5 cm, close gripper") and the same simulated robot (Franka Panda in MuJoCo):

| | **Monkey See, Monkey Do** (`monkeyseemonkeydo/`, working) | `handoff/` (in progress) |
|---|---|---|
| Input | third-person video of a person | phone video + printed ArUco marker |
| How | a vision model (GPT-6 Astra) watches the video, writes a MuJoCo scene and a robot plan, checks it in physics and fixes it until the robot succeeds | geometry: hand tracking + depth + marker give a metric wrist track, turned into per-second language actions |
| Output | demo video, presenter view, randomised **LeRobot dataset** for VLA training | language actions, VLM labels, LeRobot export, robot replay |
| Status | end to end, with a web UI | `actions`, `synth` and `replay` built; perception, VLM labels and export still to do |

Product plan and data contracts: [PRD.md](PRD.md) and [AGENTS.md](AGENTS.md).
Demo script: [monkeyseemonkeydo/DEMO.md](monkeyseemonkeydo/DEMO.md).

## Quick start

```bash
pip install -r requirements.txt              # handoff package (partner pipeline)
pip install -r monkeyseemonkeydo/requirements.txt    # monkeyseemonkeydo + web UI
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git   # Franka Panda model
echo OPENAI_API_KEY=sk-... > monkeyseemonkeydo/.env  # never commit this

cd monkeyseemonkeydo && python server.py             # web UI: http://127.0.0.1:8765
```

In the UI: pick a video (or a built-in sample), a robot and a model, press **Make the robot
do it**, then **Present** for the stage view and **Make training data** for a dataset.

The partner pipeline runs per take:

```bash
python -m handoff.synth --take takes/synth                      # fake 20 s wrist track
python run.py takes/synth --from actions --to replay            # language actions -> robot replay
pytest                                                          # 15 tests
```

## What's in the repo

### Monkey See, Monkey Do (`monkeyseemonkeydo/`): video to simulation to dataset

1. **Watch.** Frames from the video (and the video itself for Gemini) go to a vision model.
2. **Plan.** It writes the scene (primitive shapes or realistic scanned objects, sized and
   placed in metres) and a robot plan of language actions, each with a one-line physics reason.
3. **Simulate.** MuJoCo runs the plan with real contact physics: the gripper has to actually
   hold the object. We measure reach, tracking error, what is held, and task success checks.
4. **Compare and fix.** The model sees the simulation next to the video plus the
   measurements and revises, until the checks pass (usually 1 to 3 rounds, about 20 s each).
5. **Dataset.** `dataset.py` re-simulates the final plan with domain randomisation (object
   positions, rotations, colours, masses, friction, camera) and keeps the episodes that
   still succeed, as a LeRobot dataset with front + wrist cameras, joint state, actions,
   the task and the per-step language action.

Details, file map and commands: [monkeyseemonkeydo/README.md](monkeyseemonkeydo/README.md).

**Workbench UI.** The web UI at `/` is a three-step guided flow, not a single long page:

1. **Set up** — pick a video (sample or your own), a robot and a model.
2. **Run** — a short pipeline-diagram transition, then the live view: a five-stage indicator
   (watch, plan, simulate, compare & fix, result) that updates in real time, the verdict for
   the current round in plain English, your video next to the simulation, every round's
   outcome, and the robot's full plan as a telemetry-style table (target, miss in mm, what
   it's holding). A **Next: make training data** button appears the moment all checks pass.
3. **Training data** — what standard is used (LeRobot v2.1) and why, then the dataset
   generator and its results.

Completed steps are marked with a checkmark and stay clickable, so you can jump back to
review a past round or forward to training data without losing your place. The visual
language (warm paper background, bold near-black text, a single deep-amber signal colour
for anything live, monospace for every number and timestamp) is deliberately styled after
lab instrumentation — oscilloscopes, servo controllers — rather than a generic dashboard,
since this is a diagnostic tool for a simulation loop, not a consumer app. Dark mode exists
but is opt-in only, so the look doesn't depend on the presenter's OS setting during a demo.

### `handoff/`: the geometry pipeline

- `actions.py`: wrist track to EgoLAP language actions per 1 s window (unit tested).
- `synth.py`: synthetic wrist track for building the rest before perception exists.
- `replay.py`: the Franka carries out a take's language actions in MuJoCo with physics and
  checks that the object ends up where the person put it. Uses the Monkey See, Monkey Do controller.
- Still to build (see PRD): `perception.py` (MediaPipe hand landmarks + Depth Anything +
  ArUco), `transcribe.py`, `vlm.py`, `export.py`, `compose.py`, `app.py`.

### `qarm/`

A standalone QArm-style arm with slider and drag control (`python qarm/run_qarm.py
qarm/qarm.xml`). monkeyseemonkeydo also supports it as `--robot qarm`.

## Models and robots

- Models: `gpt-6-astra` (default; OpenAI's most capable), `gpt-6.1-sol` (faster, about 5x
  cheaper), `gemini-4-argon`, `claude-opus-5-5`, or `mock` (no API call). Keys are read
  from `monkeyseemonkeydo/.env` or the environment.
- Robots: Franka Panda (MuJoCo Menagerie, 7 joints, grips from above, most reliable) and a
  QArm lookalike (4 joints, grips from the side).

## Results so far

- Sample clips: GPT-6 Astra reproduced the task in 2 to 3 rounds; the robot succeeded every
  round. On the scanned-object sample it chose the scanned bowl correctly but picked a
  different toy than the one in the video.
- Dataset: 50 randomised episodes (8,066 frames) in 94 s with a 100% success rate; loads in
  LeRobot 0.6.1 after its built-in v2.1 to v3.0 conversion and yields training batches.
- Partner pipeline: the synthetic take's language actions, executed by the Franka, place
  the object 1.6 cm from where the hand released it.
- Not yet tested: a real phone recording of a person, and training a policy on the data.

## Related work

| Project | What it does | How we differ |
|---|---|---|
| [Agentic Real2Sim](https://github.com/agentic-real2sim/agentic_real2sim) (Jul 2026) | VLM agents turn a recorded **robot** episode into a MuJoCo twin | Closest competitor. Needs Linux, an NVIDIA GPU, a heavy perception stack (~80 GB) and depth; no dataset export. We start from a human RGB video, run on a laptop and export LeRobot data. |
| [Real2Render2Real](https://github.com/uynitsuj/real2render2real) | Object scan + human demo video into many robot demos (IsaacLab, kinematic) | Closest in spirit; heavy, no physics, no LeRobot output |
| [MimicGen](https://github.com/NVlabs/mimicgen), [DemoGen](https://github.com/TEA-Lab/DemoGen) | One demo into many by re-targeting segments to new object poses | Our dataset randomisation applies the same idea in MuJoCo (MimicGen's code is non-commercial, so we don't use it) |
| [RoboGen](https://github.com/Genesis-Embodied-AI/RoboGen), [GenSim](https://github.com/liruiw/GenSim) | An LLM writes simulation tasks and scenes | No video input |
| [Phantom](https://github.com/MarionLepert/phantom) | Paints a robot arm over the human in the video | Image editing, not simulation |
| [EgoMimic](https://github.com/SimarKareer/EgoMimic) | Co-trains on egocentric human video and robot data | Needs Aria glasses and an ALOHA robot |
| [EgoLAP](https://arxiv.org/html/2610.08726) | Language actions + motion reasoning for human/robot co-training | We generate data in its language-action format |
| [Gemini 4 Argon for robotics](https://developers.googleblog.com/gemini-4-argon-for-robotics/) | Closed model doing real2sim from photos and video | Our loop is open and model-agnostic |

## Third-party assets

- Franka Panda model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (cloned, not committed).
- Scanned objects in `monkeyseemonkeydo/assets/scanned_objects/`: Google Scanned Objects, CC-BY 4.0,
  MJCF conversion by Kevin Zakka (MIT). See its [ATTRIBUTION.md](monkeyseemonkeydo/assets/scanned_objects/ATTRIBUTION.md).

This repo has no license file yet; add one (for example MIT) before publishing.
