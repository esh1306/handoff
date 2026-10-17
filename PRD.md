# Handoff — PRD & Technical Setup

Oct 17, 2026 · @Sahid

## Overview

Handoff turns a phone video of a person doing a physical task into a robot-ready, labelled training episode in under 5 minutes, on a laptop, with no glove, rig or human annotator.

**Problem.** Robot models need task-specific demonstration data. Data vendors (microagi/shift, Mecka, XDOF) pay thousands of people to film tasks, then pay again to clean, segment and label that footage. Google's Gemini 4 Argon now does the labelling, but only through a paid, closed API.

**Users.** Data-ops teams at robot-data companies who process large volumes of human footage. Secondary: integrators who film a factory task before deciding how to automate it.

**Why now.** The EgoLAP paper (Princeton, Toyota, Physical Intelligence, Oct 2026) showed that plain-text motion labels plus short "why" rationales raise real-robot task progress from 44.5% to 80.1%. Its motion labels are computed from geometry with no AI, and its rationale prompts are published. Nobody ships this as an open tool that outputs the LeRobot format teams already train on.

**What we claim on stage.** We produce correct, consistent labels from cheap footage, from first- or third-person view. We do not claim we trained a better robot today.

**References:** [EgoLAP paper](https://arxiv.org/html/2610.08726) · [Gemini 4 Argon for Robotics](https://developers.googleblog.com/gemini-4-argon-for-robotics/)

## Scope

We ship one pipeline, one page and one demo; everything below the "out" line is cut, not deferred.

**In scope (must ship)**

- Single-hand, tabletop pick-and-place tasks, 10–30 s clips, 1080p 30 fps phone video.
- First-person (phone at eye/chest level) and third-person (phone on tripod) views.
- Metric scale from one printed ArUco marker on the table.
- Per-frame wrist position + gripper open/closed, smoothed.
- Language actions per 1 s window in the EgoLAP grammar.
- VLM labels: subtasks, one-line rationale per window, quality flags.
- LeRobot-format export.
- Replay of the labelled motion on a simulated arm in MuJoCo, pre-rendered to MP4.
  - Voice in: the person says the task aloud and can narrate steps; the video's audio is transcribed into the task instruction and timestamped step hints.
- One Streamlit page tying it together.

**Stretch (only after feature freeze is met early)**

- Both hands (bimanual grammar).
- Two-phone simultaneous capture with a consistency score between views.
- ElevenLabs voice-out of the episode summary (voice in is core; voice out is optional).

**Out of scope**

- Training any robot policy or VLA.
- Real robot hardware or live control.
- True stereo depth, accounts, databases, cloud deploy.

**Demo success criteria**

1. A fresh clip goes from upload to labelled episode in under 5 minutes on the M4.
2. Language actions match the visible motion direction in 9 of 10 windows we spot-check.
3. A deliberate fumble is flagged and downweighted.
4. First- and third-person clips of the same take produce the same direction words in most windows.
5. The sim arm replay visibly follows the human path.

## Functional requirements

The pipeline is seven modules run in order; each reads the previous module's file, so any one can be swapped or cached.

| # | Module | Input | Output | Tool | Owner |
| --- | --- | --- | --- | --- | --- |
| 1 | Capture | Phone | `raw/<take>_ego.mp4`, `raw/<take>_3p.mp4` | Phone camera, ArUco marker | C |
| 2 | Perception | Video | `track.csv` | MediaPipe Hands, Depth Anything V2 Small, OpenCV ArUco | A |
| 3 | Language actions | `track.csv` | `actions.jsonl` | Our code (EgoLAP grammar) | B |
| 4 | VLM labels | Keyframes + actions | `labels.json` | OpenAI vision model | B |
| 5 | Export | All above | `lerobot/<take>/` | LeRobot | B |
| 6 | Replay | `track.csv` | `replay.mp4` | MuJoCo + mink + Menagerie Franka | C |
| 7 | UI | All above | Streamlit page | Streamlit | C |

**1. Capture**

- Print a 4x4 ArUco marker (DICT\_4X4\_50, id 0) at exactly 10 cm side; tape it flat on the table, visible all clip.
- Record 1080p, 30 fps, locked exposure if the phone allows. No zoom. Audio on: say the task aloud in the first 3 s, clap once to sync phones, and narrate steps if you like.
- Ego: phone held at eye level or clipped to a cap, pointing down at \~45 degrees. Third-person: tripod 1–1.5 m away, side-front angle.
- Calibrate each phone once with a printed chessboard (20 photos) to get intrinsics; fallback is focal length = 1.2 x image width.

**2. Perception**

- MediaPipe Hands per frame: 21 landmarks, handedness, confidence. Use the wrist (0), thumb tip (4), index tip (8), and index MCP (5).
- Depth Anything V2 Small per frame (or every 2nd frame, interpolated) for relative depth at the wrist pixel.
- ArUco pose gives the table plane and metric scale. Wrist 3D = back-project wrist pixel; depth scaled so the marker's depth matches its known pose.
- Gripper state: closed if thumb-to-index tip distance < 0.35 x hand length (wrist to index MCP), with 3-frame hysteresis.
- Smooth position with a Savitzky–Golay filter (window 9, order 2). Interpolate gaps of 5 frames or fewer; longer gaps are marked `occluded`.
- Express positions in the camera frame (matches EgoLAP for human data) and also in the marker/table frame for replay.

**3. Language actions**

- Window = 1 s (30 frames), stride 1 s.
- Delta position and delta rotation computed in the camera frame at window start.
- Serialize with the EgoLAP grammar: translation in whole cm, rotation in multiples of 5 degrees, zero components dropped, gripper state always last.
- Wrist rotation: estimate from the hand plane (wrist, index MCP, pinky MCP); if confidence is low, emit translation only and set `rot_valid=false`.

**4. VLM labels**

- Transcribe the clip's audio first (OpenAI or ElevenLabs speech-to-text, word timestamps). First sentence becomes the task instruction; later phrases become step hints with times. Send 1 keyframe per window (middle frame, 512 px) + the language action + task instruction + step hints.
- Return strict JSON: subtask plan, per-window current subtask, one-sentence rationale, quality flag (`ok`, `pause`, `fumble`, `occluded`, `retry`) and weight (1.0, 0.5, 0.2, 0).
- Prompt adapted from the paper's MECKA prompt: hand terms, no future events described as done, rationale must agree with motion direction.
- Cache responses by hash of inputs so re-runs are free. Model name lives in one config value: gpt-6-astra for final demo takes, a cheaper GPT-5.6 tier while developing (Astra lists at $10 in / $50 out per 1M tokens, about $0.50 per 20 s clip).

**5. Export**

- LeRobotDataset v2-style: `observation.images.ego` (mp4), `observation.state` (wrist xyz + 6D rot + gripper, 10 dims), `action` (next-step delta), `task`, plus our extra columns `language_action`, `rationale`, `quality`, `weight`.
- Validate by loading it back with `LeRobotDataset` and printing one frame.

**6. Replay**

- Load Franka Panda from MuJoCo Menagerie; map the table-frame wrist path into the robot workspace (fixed offset + scale clamp).
- Solve IK per frame with mink; drive gripper open/close from state.
- Place a cylinder "mug" at the object's start position; when the gripper closes near it, attach it to the hand (kinematic attach, not a friction grasp) and release on open. Render offscreen to `replay.mp4` at 30 fps, same frame count as the source clip; log frames where IK error > 2 cm as `unreachable`. On stage this is a kinematic replay: the labels, executed.

**7. UI**

- Upload or pick a cached take; one button runs the pipeline with a progress bar per module.
- Main view: real video left with the hand track and marker drawn on top, replay MP4 right, synced; below, a timeline strip coloured by quality flag. Clicking a second seeks both videos.
- Click a window to see its language action, subtask and rationale.
- Download button for the LeRobot folder (zip).

## Data formats

Three contracts let the three of us build in parallel from minute one: lock these before writing code.

**track.csv** (Module 2 output, one row per frame; positions in metres)

```csv
frame,t,cam_x,cam_y,cam_z,tbl_x,tbl_y,tbl_z,rot6d_0,rot6d_1,rot6d_2,rot6d_3,rot6d_4,rot6d_5,grip,hand_conf,occluded
0,0.000,0.012,-0.084,0.512,0.231,0.104,0.087,1,0,0,0,1,0,0,0.94,0
1,0.033,0.013,-0.083,0.509,0.232,0.104,0.085,1,0,0,0,1,0,0,0.95,0
```

`grip`: 0 = open, 1 = closed. `rot6d_*`: first two columns of the wrist rotation matrix. `occluded`: 1 if the hand was not tracked and the gap was too long to interpolate.

**actions.jsonl** (Module 3 output, one line per 1 s window)

```json
{"window": 3, "t_start": 3.0, "t_end": 4.0, "dx_cm": 5, "dz_cm": -3, "dy_cm": 0, "roll_deg": 0, "pitch_deg": 10, "yaw_deg": -5, "grip_end": "close", "rot_valid": true, "text": "Right arm: move forward 5 cm, move down 3 cm, tilt back 10 degrees, rotate clockwise 5 degrees, close gripper"}
```

**Language-action grammar** (copied from EgoLAP; order is fixed)

| Component | Positive | Negative | Rounding |
| --- | --- | --- | --- |
| Translation x | move forward n cm | move back n cm | whole cm |
| Translation z | move up n cm | move down n cm | whole cm |
| Translation y | move left n cm | move right n cm | whole cm |
| Roll | tilt left θ degrees | tilt right θ degrees | 5 degrees |
| Pitch | tilt back θ degrees | tilt forward θ degrees | 5 degrees |
| Yaw | rotate counterclockwise θ degrees | rotate clockwise θ degrees | 5 degrees |
| Gripper | open gripper | close gripper | always emitted |

Components that round to zero are omitted. Prefix is `Right arm:` or `Left arm:` by handedness. Primary output uses the table frame set by the marker (forward = away from the person, left, up), so the ego and third-person phones produce the same words; a camera-frame copy is also saved to match EgoLAP's human-data convention. Print the marker with a FORWARD arrow on its top edge and place it pointing away from the person.

**labels.json** (Module 4 output)

```json
{
  "task": "put the mug in the bowl",
  "subtasks": ["reach mug", "grasp mug", "lift and carry", "place in bowl", "release"],
  "windows": [
    {"window": 3, "subtask": "grasp mug", "rationale": "The open hand is beside the mug handle; closing the fingers while descending secures the handle before lifting.", "quality": "ok", "weight": 1.0}
  ],
  "summary": "One clean pick-and-place with one regrasp at 6 s."
}
```

**Repo layout**

```
handoff/
  raw/            # phone videos (gitignored)
  takes/<take>/   # track.csv, actions.jsonl, labels.json, replay.mp4, keyframes/
  export/<take>/  # LeRobot dataset
  handoff/
    perception.py # Module 2
    actions.py    # Module 3
    vlm.py        # Module 4
    export.py     # Module 5
    replay.py     # Module 6
  app.py          # Module 7 (Streamlit)
  run.py          # python run.py takes/<take> runs 2→6
  assets/marker_0.png, calib/
```

## Technical setup

Everything runs on the MacBook M4 Air; the ASUS TUF A15 is a second worker for heavy perception and rendering, not a requirement.

| Machine | Role | Runs |
| --- | --- | --- |
| MacBook M4 Air | Main dev + demo machine | Whole pipeline, Streamlit, MuJoCo replay; depth on Apple GPU (MPS) |
| ASUS TUF A15 (NVIDIA) | Heavy worker | Depth on CUDA for long clips; optional WiLoR hand mesh; batch renders |
| Phones (2) | Capture | Ego + third-person video; chessboard calibration photos |

**API keys**

- `OPENAI_API_KEY` from the hackathon credits (required: labels + transcription). Confirm at the desk that the credits can call gpt-6-astra.
- `ELEVENLABS_API_KEY` (optional, voice-over).
- Gemini is not needed.

**1. Environment (both laptops)**

```bash
# Python 3.11 keeps MediaPipe and LeRobot happy
brew install python@3.11 ffmpeg git   # Mac
# Windows/Linux on the ASUS: install Python 3.11 + ffmpeg, or use conda

python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

pip install mediapipe opencv-contrib-python numpy scipy pandas
pip install torch torchvision transformers pillow   # Depth Anything V2 via HF
pip install mujoco mink                             # sim + IK
pip install lerobot                                 # dataset export
pip install streamlit openai python-dotenv tqdm

git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
```

On the ASUS, install the CUDA build of torch from pytorch.org instead of the default wheel.

**2. Smoke tests (run all five before writing pipeline code; 15 min total)**

```bash
# MediaPipe imports and finds a hand in a webcam frame
python -c "import mediapipe as mp; print(mp.__version__)"

# Depth model loads on the Apple GPU
python -c "import torch; print(torch.backends.mps.is_available())"
python -c "from transformers import pipeline; p=pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf'); print('ok')"

# MuJoCo loads the Franka from Menagerie
python -c "import mujoco; m=mujoco.MjModel.from_xml_path('mujoco_menagerie/franka_emika_panda/scene.xml'); print(m.nq)"

# ArUco detector exists
python -c "import cv2; print(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))"

# LeRobot imports
python -c "import lerobot; print(lerobot.__version__)"
```

On Mac, MuJoCo's interactive viewer needs `mjpython` instead of `python`; offscreen rendering for `replay.mp4` works with plain `python`.

**3. Marker and calibration (Person C, first 20 min)**

```bash
python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50); cv2.imwrite('assets/marker_0.png', cv2.aruco.generateImageMarker(d, 0, 800))"
```

Print it so the black square measures exactly 10 cm; measure with a ruler and record the real size in `calib/marker.json`. Print an OpenCV 9x6 chessboard, take 20 photos per phone, and run `cv2.calibrateCamera` to save `calib/<phone>.json`.

**4. First test clip**

Record one 20 s take of "put the mug in the bowl" from both phones before any code is merged. Every module develops against this clip.

**5. Git**

One repo, one branch per person (`perception`, `labels`, `demo`), merge to `main` at each checkpoint. Large videos stay out of git; share them by AirDrop or a shared folder.

## Team and timeline

Five one-hour blocks with a hard checkpoint at the end of each; if a checkpoint is missed, cut scope rather than slide the next one.

| Person | Owns | Machine | First task |
| --- | --- | --- | --- |
| A | Perception (Module 2) | ASUS, then M4 | MediaPipe wrist track on the test clip |
| B | Labels + export (Modules 3–5) | M4 | Grammar serializer on a fake `track.csv` |
| C | Capture, replay, UI (Modules 1, 6, 7) | M4 | Marker, calibration, test clip; then Franka following a circle |

1. **Hour 1: parts work alone.** A outputs a 2D wrist track + grip state on the test clip. B turns a hand-written `track.csv` into correct language actions. C has the marker, calibration, test clip from both phones, and the Franka tracing a circle in MuJoCo.
   - Checkpoint: each person shows their output on screen.
2. **Hour 2: real 3D track, real actions.** A adds depth + marker scale and writes `track.csv`. B runs the serializer on A's file. C drives the Franka from A's file.
   - Checkpoint: language actions for the test clip read correctly against the video.
3. **Hour 3: VLM labels and export.** B adds subtasks, rationales and quality flags, then the LeRobot export. C builds the Streamlit page with video, replay and timeline.
   - Checkpoint: `python run.py takes/test` produces every file.
4. **Hour 4: integrate, then freeze.** Whole flow runs from the UI. Fix only what breaks the demo.
   - Checkpoint: feature freeze at the end of this hour. Anything not working is hidden.
5. **Hour 5: record and rehearse.** Film the final ego + third-person takes with a deliberate fumble. Pre-process them as the fallback. Rehearse the 3-minute demo twice with a timer.
   - Checkpoint: two clean runs of the demo under 3 minutes.

**If behind**

- End of hour 2 with no 3D track: drop depth, use the marker plane only (assume the hand moves at table height + fixed offset).
- End of hour 3 with no replay: show the 3D path as a plotted line instead of the arm.
- End of hour 3 with flaky VLM output: keep language actions and quality flags, drop rationales.

## Demo, risks and fallbacks

The demo runs on pre-processed takes by default; the live upload is a bonus we start only if time is left.

**3-minute script**

1. (0:00–0:30) Problem: robot companies pay thousands of people to film tasks, then pay again to label them. Google just put that labelling behind a paid API.
2. (0:30–1:00) Teammate does "put the mug in the bowl" live, filmed by two phones, with one deliberate fumbled grasp.
3. (1:00–2:00) Open the processed take: video and sim arm side by side, timeline below. Click through windows: language action, subtask, rationale. The fumble window is red and weighted 0.2.
4. (2:00–2:30) Switch to the third-person take: the same direction words appear in the same windows.
5. (2:30–3:00) Download the LeRobot folder. Close: open, local, phone-only, and in the format robot teams already train on.

**Risks**

| Risk | Likelihood | Fallback |
| --- | --- | --- |
| Monocular depth jitter makes cm values noisy | High | Smoothing + the grammar's 1 cm rounding; anchor depth to the marker |
| Hand hidden behind the mug during grasp | High | Interpolate short gaps; flag long ones as `occluded` (shown as a feature) |
| Wrist rotation estimate unreliable | Medium | Emit translation only and mark `rot_valid=false` |
| VLM returns invalid JSON or wrong directions | Medium | Strict JSON schema, retry once, check rationale direction against the action |
| Human path out of the Franka's reach | Medium | Scale and offset the path into the workspace; log `unreachable` frames |
| Depth too slow on the M4 | Low | Run every 2nd frame, or process on the ASUS |
| Venue Wi-Fi fails during the live demo | Medium | All takes and VLM responses cached locally |

**Questions judges will ask**

- "Isn't this Argon?" Argon is closed and paid. Our motion labels come from geometry, not a model guess, and the output is LeRobot.
- "Does it make robots better?" The EgoLAP paper measured that these labels do; we make producing them cheap. We did not train a policy today.
- "Why should a data company trust it?" Every window shows its confidence and quality flag, and the sim replay is a visual check of the label.
