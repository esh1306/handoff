# video2sim

Video of a person doing a tabletop task in; MuJoCo simulation of a robot doing the same
task, and a LeRobot training dataset, out.

A vision model (GPT-6 Astra by default) watches the video and writes the scene plus a robot
plan; MuJoCo runs it with real physics and measures the result; the model compares the
simulation with the video and fixes the plan until the task succeeds. This is the real2sim
loop from the Gemini 4 Argon post, with EgoLAP-style language actions ("move down 12 cm,
close gripper") and a one-line reason per step. Then one video becomes many training
episodes through domain randomisation.

## Setup

```bash
pip install -r video2sim/requirements.txt
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git   # repo root or video2sim/assets/
echo OPENAI_API_KEY=sk-... > video2sim/.env     # never commit this file
```

## Use

```bash
cd video2sim
python server.py                  # web UI at http://127.0.0.1:8765  (--port to change)
python loop.py --video clip.mp4 --robot panda --model gpt-6-astra   # same loop from the CLI
python dataset.py runs/<id> --episodes 50                            # training dataset
python replay.py runs/<id>        # interactive 3D viewer of a run
python make_sample.py [--toy]     # re-render the sample clips
```

### Web UI

- **Workbench** (`/`): pick a video (your own or a sample), robot and model; live progress
  through the five stages; verdict in plain words; your video and the robot side by side;
  what changed each round; the robot's plan with reasons; raw model answers and logs.
  Phone videos (HEVC `.mov`, rotated) are converted to H.264 on upload.
- **Presenter view** (`/demo#<run id>`, the **Present** button): full-screen person vs robot,
  with the robot's language actions highlighted as its video plays. See [DEMO.md](DEMO.md).
- **Training data**: generate a dataset for a finished run, watch progress, preview
  episodes, download a zip.

### Training dataset

`dataset.py` re-simulates the run's final plan with randomised object positions (+/-4 cm),
rotations (+/-20 deg), colours, masses, friction and camera angle. The plan's move targets
follow the moved objects (the grasped object until the gripper closes, the destination
after), and only episodes where the robot still succeeds are kept. Output is the LeRobot
v2.1 layout in `datasets/<run id>/`:

| Feature | Contents |
|---|---|
| `observation.images.front`, `observation.images.wrist` | 256x256 RGB video, 15 fps |
| `observation.state` | joint angles + gripper (0 open .. 1 closed) |
| `observation.ee_pos` | tool point position (m) |
| `action` | absolute joint targets + gripper command (0/1) |
| `language_action`, `reason` | the current plan step's language action and the model's reason |
| `task_index` | the task instruction |

Current LeRobot (0.6+) reads v3.0. Convert once, then load:

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=local/<name> --root=<dataset folder> --push-to-hub=false
```

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/<name>", root="<dataset folder>", video_backend="pyav")
```

Train, for example, ACT (`lerobot-train --policy.type=act --dataset.repo_id=local/<name>
--dataset.root=<dataset folder> ...`, check flags against your LeRobot version). Images are
clean renders, so for real robots mix with real data.

### Scanned objects

Besides boxes, cylinders, spheres and open containers, the model can place nine realistic
scanned objects (`"shape": "mesh", "mesh": "<key>"`, listed in `assets/objects.json`): five
graspable items narrower than the gripper (triceratops and Android toys, a rhino figure, a
bottle box, an ink cartridge box), three bowls and a plate. Each uses the textured scan for
rendering and a 32-piece convex decomposition for contacts; the catalogue records a grasp
(gripper yaw and height) that worked in testing. Source and licences:
[assets/scanned_objects/ATTRIBUTION.md](assets/scanned_objects/ATTRIBUTION.md).

### Models and robots

- Models: `gpt-6-astra`, `gpt-6.1-sol`, `gemini-4-argon`, `claude-opus-5-5`, or `mock`
  (canned answer, no API call). The provider is picked from the model name.
- Robots: `panda` (Franka, grips from above) or `qarm` (4-DOF QArm lookalike, grips from the
  side). Both have a wrist camera.

## Files

| File | What it does |
|---|---|
| `loop.py` | The analyse, simulate, critique loop; writes `runs/<id>/` and a live `events.jsonl` |
| `server.py`, `static/index.html`, `static/demo.html` | Web UI: workbench, presenter view, dataset generation |
| `providers.py` | Swappable model backends (OpenAI, Gemini, Anthropic, mock) |
| `prompts.py` | What the model is asked, including the scanned-object catalogue |
| `scene.py` | JSON scene spec to MuJoCo model (primitives and scanned meshes) |
| `controller.py` | IK, plan execution, video recording, success checks |
| `dataset.py` | Domain-randomised re-simulation to a LeRobot dataset |
| `robots.py` | Panda and QArm adapters (joints, gripper, wrist camera) |
| `replay.py` | Interactive viewer |
| `make_sample.py` | Renders the sample clips in `samples/` |
| `DEMO.md` | How to record, process and present a demo |

Run outputs (`runs/`), datasets (`datasets/`) and `.env` are gitignored.
