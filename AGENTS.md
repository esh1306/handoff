# AGENTS.md — Handoff

> Put this file at the root of your repo. Cursor reads `AGENTS.md` automatically.
> Then paste the **First prompt** (bottom of this file) into Cursor's agent.

## What we are building (one sentence)

Handoff turns a phone video of a person doing a tabletop task into a robot-ready,
labelled training episode: wrist track, gripper state, per-second "language actions"
(EgoLAP grammar), VLM subtasks/rationales/quality flags, a LeRobot-format export, and a
MuJoCo replay of the motion on a Franka arm. Hackathon build: ~5 hours, 3 people,
MacBook M4 Air (main) + ASUS TUF A15 with NVIDIA GPU (helper). No robot hardware.

## Ground rules for the agent

- Python 3.11. Every module is a plain function + a CLI; no frameworks beyond those listed.
- Device order for torch: `cuda` → `mps` → `cpu`. Never assume CUDA.
- Each module reads files from `takes/<take>/` and writes files there. Modules never call
  each other directly, so any one can be rerun or replaced.
- Never block on network: cache every API response in `takes/<take>/cache/` keyed by a
  hash of its inputs.
- When a library API is uncertain (LeRobot, mink, MediaPipe Tasks, OpenAI SDK), inspect
  the installed package source/docs first and adapt. Do not invent function names.
- Prefer working and ugly over elegant and unfinished. Log, don't crash: a failed frame
  becomes a flagged frame.
- Write a tiny test or smoke script for each module that runs on synthetic data.

## Repo layout

```
handoff/
  AGENTS.md
  .env.example          # OPENAI_API_KEY=, ELEVENLABS_API_KEY=
  config.yaml            # all tunables (below)
  requirements.txt
  run.py                 # python run.py takes/<take> [--from perception] [--to compose]
  app.py                 # Streamlit UI
  handoff/
    __init__.py
    config.py            # loads config.yaml + .env
    capture.py           # marker generation, camera calibration, audio extraction
    perception.py        # video -> track.csv
    transcribe.py         # audio -> transcript.json
    actions.py            # track.csv -> actions.jsonl (EgoLAP grammar)
    vlm.py                 # keyframes + actions + transcript -> labels.json
    export.py              # -> export/<take>/ (LeRobot)
    replay.py              # track.csv -> replay.mp4 (MuJoCo, kinematic)
    compose.py             # source + replay + timeline -> demo.mp4
    synth.py               # synthetic track.csv generator for testing
  assets/marker_0.png
  calib/                 # <phone>.json intrinsics, marker.json
  raw/                   # input videos (gitignored)
  takes/<take>/          # all per-take outputs
  export/<take>/
  mujoco_menagerie/      # git clone, gitignored
  tests/
```

## config.yaml (create with these defaults)

```yaml
fps: 30
window_s: 1.0
marker:
  dict: DICT_4X4_50
  id: 0
  size_m: 0.10          # measured printed black-square size
camera:
  calib_file: calib/iphone.json   # fallback: fx = fy = 1.2 * width, cx/cy = centre
perception:
  depth_model: depth-anything/Depth-Anything-V2-Small-hf
  depth_every_n: 2
  grip_close_ratio: 0.35          # thumb-index tip dist / hand length
  grip_hysteresis_frames: 3
  max_interp_gap: 5
  savgol_window: 9
  savgol_order: 2
vlm:
  model_final: gpt-6-astra
  model_dev: gpt-5.6-luna
  use_final: false
  keyframe_px: 512
transcribe:
  model: whisper-1
replay:
  menagerie_scene: mujoco_menagerie/franka_emika_panda/scene.xml
  robot_base_in_table: [0.0, -0.45, 0.0]   # metres; robot sits where the person sits
  max_reach_m: 0.85
  attach_dist_m: 0.06
  render_w: 960
  render_h: 720
```

## Data contracts (do not change without telling the team)

### track.csv — one row per frame

```
frame,t,cam_x,cam_y,cam_z,tbl_x,tbl_y,tbl_z,rot6d_0..rot6d_5,grip,hand_conf,occluded,handedness
```

- `cam_*`: wrist position in the camera optical frame (x right, y down, z forward), metres.
- `tbl_*`: wrist position in the TABLE frame, metres. Table frame = marker frame:
  origin at marker centre, +x = forward (away from person), +y = left, +z = up.
- `rot6d_*`: first two columns of the wrist rotation matrix, in the table frame.
- `grip`: 0 open, 1 closed. `occluded`: 1 if untracked and gap too long to interpolate.

### actions.jsonl — one line per window

```json
{"window": 3, "t_start": 3.0, "t_end": 4.0,
 "frame": "table",
 "dx_cm": 5, "dy_cm": 0, "dz_cm": -3, "roll_deg": 0, "pitch_deg": 10, "yaw_deg": -5,
 "grip_end": "close", "rot_valid": true, "occluded_frac": 0.0,
 "text": "Right arm: move forward 5 cm, move down 3 cm, tilt back 10 degrees, rotate clockwise 5 degrees, close gripper",
 "text_camera_frame": "Right arm: ..."}
```

### transcript.json

```json
{"text": "...", "task": "put the mug in the bowl",
 "words": [{"w": "grab", "t0": 4.1, "t1": 4.3}],
 "hints": [{"t": 4.1, "phrase": "grabbing it"}]}
```

### labels.json

```json
{"task": "put the mug in the bowl",
 "subtasks": ["reach mug", "grasp mug", "lift and carry", "place in bowl", "release"],
 "windows": [{"window": 3, "subtask": "grasp mug",
              "rationale": "The open hand is beside the handle; closing while descending secures it before lifting.",
              "quality": "ok", "weight": 1.0}],
 "summary": "One clean pick-and-place with one regrasp at 6 s.",
 "model": "gpt-4o"}
```

`quality` ∈ `ok | pause | fumble | occluded | retry`; default weights 1.0, 0.5, 0.2, 0.0, 0.2.

## EgoLAP language-action grammar (copy exactly)

Order inside one clause: translation x, z, y, then roll, pitch, yaw, then gripper.

| Component | Positive | Negative | Rounding |
|---|---|---|---|
| Translation x (forward) | `move forward n cm` | `move back n cm` | integer cm |
| Translation z (up) | `move up n cm` | `move down n cm` | integer cm |
| Translation y (left) | `move left n cm` | `move right n cm` | integer cm |
| Roll (about forward) | `tilt left θ degrees` | `tilt right θ degrees` | nearest 5° |
| Pitch (about left) | `tilt back θ degrees` | `tilt forward θ degrees` | nearest 5° |
| Yaw (about up) | `rotate counterclockwise θ degrees` | `rotate clockwise θ degrees` | nearest 5° |
| Gripper (final state) | `open gripper` | `close gripper` | always emitted |

- Components that round to zero are omitted. Join with `, `. Prefix `Right arm: ` or
  `Left arm: ` from handedness.
- Rotation delta over the window: `R_rel = R_end @ R_start.T` (table frame), converted to
  extrinsic XYZ Euler (scipy `Rotation.from_matrix(...).as_euler("xyz", degrees=True)`),
  mapped roll=x, pitch=y, yaw=z.
- If `rot_valid` is false, emit translation + gripper only.
- Also produce `text_camera_frame` using camera axes mapped as
  forward = +z_cam, left = −x_cam, up = −y_cam (EgoLAP's human convention).
- Unit-test the serializer with hand-made deltas, including all-zero (→ only gripper),
  negative values, and rounding edges (0.49 cm → omitted, 2.5° → 5° or 0°, be consistent).

## Module specs

### capture.py
- `make_marker(out="assets/marker_0.png")`: 800 px DICT_4X4_50 id 0, with a white border
  and the text "FORWARD ↑" above the top edge (use PIL). Users print it at exactly 10 cm.
- `calibrate(images_glob, board=(9,6), square_m=0.025, out="calib/<name>.json")`.
- `extract_audio(video, out_wav)` via ffmpeg (16 kHz mono).
- `sync_offset(video_a, video_b)`: find the clap as the largest audio onset in the first
  10 s; return offset seconds. (Stretch; used for two-phone triangulation.)

### perception.py — the critical path, build first
1. Read video with OpenCV; if larger than 1280 px wide, downscale for processing (keep
   intrinsics consistent by scaling fx, fy, cx, cy).
2. Hands: MediaPipe **Tasks** `HandLandmarker` (VIDEO running mode). Download
   `hand_landmarker.task` from
   `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
   into `assets/`. Use the highest-confidence hand (or the configured handedness). Keep
   both image landmarks (pixels) and world landmarks (metres, hand-centred).
3. Marker: `cv2.aruco.ArucoDetector`; pose via `cv2.solvePnP` with object points
   `[-L/2, L/2, 0], [L/2, L/2, 0], [L/2, -L/2, 0], [-L/2, -L/2, 0]` (TL, TR, BR, BL).
   Marker axes (OpenCV): x right, y up (towards "FORWARD"), z out of table. Convert to our
   table frame: forward = +y_marker, left = −x_marker, up = +z_marker. If the camera is
   static (third person), estimate the pose once from the median of the first 30 frames.
   If moving (ego), estimate per frame and interpolate missing detections.
4. Wrist depth (two estimators, combine):
   - **A. Depth model + table-plane fit.** Run Depth Anything V2 Small every N frames
     (output is relative inverse depth). Sample ~200 pixels on the table around the
     marker (in a ring 1–3 marker-widths away, excluding the hand's bounding box). For
     those pixels compute true depth by intersecting the pixel ray with the table plane.
     Fit `1/z_true ≈ a * d_model + b` by least squares (RANSAC if noisy). Wrist depth =
     `1 / (a * d_wrist + b)`, where `d_wrist` is the median model value in a 5×5 patch.
   - **B. Hand-size PnP.** `cv2.solvePnP` between the 21 world landmarks and 21 image
     landmarks gives the hand's translation in camera frame from its known metric size.
   - Use A when its fit residual is low; otherwise B; record which in a debug column. If
     both are bad for a frame, mark low confidence.
5. Back-project wrist pixel with depth → `cam_xyz`; transform to `tbl_xyz`.
6. Rotation: build the hand frame from world landmarks wrist(0), index_mcp(5),
   pinky_mcp(17): x = normalize(index_mcp − pinky_mcp), z = normalize(cross(x,
   middle_mcp(9) − wrist)), y = cross(z, x). Rotate camera→table. `rot_valid` = hand_conf
   above threshold.
7. Grip: closed if `dist(thumb_tip 4, index_tip 8) < ratio * dist(wrist 0, index_mcp 5)`,
   with hysteresis. Use world landmarks for scale-free distance.
8. Gaps ≤ `max_interp_gap` frames: linear interpolation; longer: `occluded=1`.
   Savitzky–Golay smoothing on positions (skip occluded spans).
9. Also write `takes/<take>/overlay.mp4`: source with hand skeleton, wrist dot, marker
   axes and a GRIP OPEN/CLOSED badge. Used by compose.py.
10. CLI: `python -m handoff.perception raw/test_3p.mp4 --take takes/test --calib calib/iphone.json`.

### transcribe.py
- Extract audio, call OpenAI `audio.transcriptions.create(model="whisper-1",
  response_format="verbose_json", timestamp_granularities=["word"])`.
- `task` = first sentence (strip fillers like "okay", "so"). `hints` = remaining phrases
  with their start times. Empty audio → empty transcript, pipeline continues.

### actions.py
- Windows of `window_s` from t=0. Delta = last valid frame − first valid frame in window.
- `occluded_frac` = fraction of occluded frames in window.
- Pure functions, fully unit-tested (`tests/test_actions.py`).

### vlm.py
- Keyframe = middle frame of each window from `overlay.mp4` (resize to `keyframe_px`),
  JPEG, base64.
- One request per take (all windows) if it fits, else batches of 10 windows. Use the
  OpenAI Responses API with image inputs and a strict JSON schema (structured outputs)
  matching `labels.json`. Verify the exact SDK call shape in the installed `openai`
  package before writing it.
- System prompt (adapt from EgoLAP's MECKA prompt): label first-person or third-person
  human manipulation; use hand terms; ground claims in visible evidence; never describe a
  future grasp or release as already done; the rationale must agree with the given motion
  direction; flag pauses, fumbles, retries and hidden hands; return JSON only.
- Post-check: if a rationale says "up"/"lift" while `dz_cm < 0` (or the reverse), retry
  that window once; if still wrong, keep the label and set quality to `ok` but mark
  `rationale_checked=false`.
- If `occluded_frac > 0.5`, force quality `occluded`.
- Model = `model_final` if `use_final` else `model_dev`.

### export.py
- Use `lerobot`'s `LeRobotDataset.create(...)` → `add_frame(...)` → `save_episode()`.
  Inspect the installed version's signatures first. Features:
  `observation.images.cam` (video), `observation.state` float32[10] (tbl xyz + rot6d +
  grip), `action` float32[10] (next state − state, grip as target), plus per-frame string
  columns `language_action`, `rationale`, `quality` and float `weight` (frames inherit
  their window's labels). `task` = transcript task or CLI fallback.
- If the LeRobot API fights you for more than 20 minutes: write the v2 layout by hand
  (`data/chunk-000/episode_000000.parquet`, `videos/...`, `meta/info.json`,
  `meta/episodes.jsonl`, `meta/tasks.jsonl`) and note it in the README.
- Validate by loading it back and printing frame 0.

### replay.py — kinematic, deterministic, no physics stepping
- Build the scene with `mujoco.MjSpec`: load the Menagerie Panda `scene.xml`, add a table
  box, a cylinder "mug" (r 4 cm, h 9 cm, freejoint) and a bowl marker (flat disc).
  Set `model.vis.global_.offwidth/offheight` ≥ render size before creating the renderer.
- Map table frame → robot base frame: robot base at `robot_base_in_table`, facing +x.
  `p_robot = p_tbl − base`, then clamp radius to `max_reach_m`.
- Mug start position = wrist position at the first close event, projected to table height.
- IK with `mink`: `Configuration` + `FrameTask` on the Panda's end-effector site
  (inspect the model for the site name, e.g. an attachment/grip site), position cost 1.0,
  orientation cost 0.1 with a top-down target rotated by the hand's yaw; posture task for
  stability; solve with daqp or quadprog (whatever is installed); warm-start from the
  previous frame. Log IK error per frame; > 2 cm → `unreachable`.
- Gripper: set finger joints open/closed from `grip`.
- Kinematic attach: when grip closes and mug is within `attach_dist_m`, store the
  hand→mug offset; while attached, set the mug freejoint qpos = hand pose × offset; on
  open, detach and drop the mug to table height. Call `mj_forward` each frame (no
  `mj_step`).
- Render with `mujoco.Renderer`, a fixed camera at ~45° front-side, one frame per source
  frame; write `replay.mp4` with `imageio` (ffmpeg). Same frame count and fps as source.

### compose.py — the demo video (robust, no browser sync needed)
- ffmpeg: `overlay.mp4` (left) | `replay.mp4` (right), same height, `hstack`.
- Below, a 120 px strip: coloured timeline (green ok, amber pause/retry, red fumble, grey
  occluded), a moving playhead, and burned-in text for the current window: language
  action (line 1), subtask + rationale (line 2). Render the strip frames with PIL, encode
  with imageio, `vstack` with ffmpeg.
- Output `takes/<take>/demo.mp4`.

### app.py (Streamlit)
- Sidebar: choose an existing take or upload a video (+ phone calib choice). "Run" button
  runs `run.py` stages with `st.status` per stage; skip stages whose outputs exist unless
  "force".
- Main: `st.video(demo.mp4, start_time=...)`. Under it a dataframe of windows (time,
  language action, subtask, quality, weight); selecting a row sets `start_time`.
- Tabs: "Episode JSON", "3D path" (plotly 3D line of tbl_xyz coloured by grip),
  "Download" (zip of `export/<take>/`).
- If two takes share a prefix (`<take>_ego`, `<take>_3p`), show a "Compare views" tab:
  per-window agreement of direction words, overall % match.

### run.py
`python run.py takes/<take> --video raw/<file>.mp4 [--from X] [--to Y] [--force]`
Stages in order: perception, transcribe, actions, vlm, export, replay, compose.

### synth.py
Generate a fake 20 s `track.csv` (reach → grasp → lift 15 cm → move left 20 cm → lower
→ release) so actions/vlm/export/replay can be built before perception works.

## requirements.txt

```
mediapipe
opencv-contrib-python
numpy
scipy
pandas
pillow
torch
torchvision
transformers
mujoco
mink
lerobot
imageio[ffmpeg]
streamlit
plotly
openai
python-dotenv
pyyaml
tqdm
pytest
```

System: `ffmpeg`. On Mac: `brew install python@3.11 ffmpeg`. On the ASUS: CUDA build of
torch from pytorch.org. Clone `https://github.com/google-deepmind/mujoco_menagerie`.

## Build order and owners

| Order | Task | Owner | Done when |
|---|---|---|---|
| 1 | Scaffold repo, config, requirements, synth.py, smoke tests | anyone, first 20 min | `pytest` passes, smoke tests print OK |
| 2 | capture.make_marker + calibrate | C | marker printed at 10 cm; calib json saved |
| 3 | actions.py + tests (on synth track) | B | serializer tests pass |
| 4 | perception.py 2D hand + grip + overlay | A | overlay.mp4 looks right on test clip |
| 5 | replay.py on synth track | C | Franka follows synth path, mug attaches |
| 6 | perception 3D (marker + depth + PnP) → track.csv | A | path plot matches the real motion |
| 7 | transcribe.py + vlm.py | B | labels.json valid on test clip |
| 8 | export.py | B | dataset loads back |
| 9 | compose.py + app.py | C | demo.mp4 plays in Streamlit |
| 10 | Freeze, record final takes, cache everything | all | two clean demo runs |

Stretch only after step 10: two-phone triangulation using marker poses from both cameras
+ clap sync; ElevenLabs voice-out of `summary`; bimanual.
