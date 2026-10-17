# Demo flow: record a clip, show the robot doing it

## 1. Record (5 minutes, do it before the demo)

- **Task:** one hand picks up one small object and puts it in or on something.
  Good: a coloured block or a lemon into a bowl, a small box onto a plate.
  The object must be narrower than 8 cm (the Franka's gripper opening).
- **Camera:** phone on a tripod or a stack of books, 1 to 1.5 m away, side-on and a
  little above the table. Landscape. Don't touch it while recording.
- **Table:** plain surface, good even light, nothing else on it. Avoid shiny,
  see-through or very dark objects.
- **Motion:** hand starts out of frame or resting, reach, grasp, lift, carry, place,
  release, pull the hand back. Slow and clear. 10 to 20 s in total.
- Record 2 or 3 takes; keep the cleanest.

## 2. Process it (about 1 to 2 minutes per clip)

```bash
cd video2sim
python server.py            # http://127.0.0.1:8765
```

In the workbench: **My own video**, choose the clip, **Franka Panda**, **gpt-6-astra**,
type the task under More options ("puts the lemon in the bowl"), then **Make the robot
do it**. Do this before going on stage so the result is cached; runs stay in
`video2sim/runs/` and need no network to show again.

If a take fails, record another rather than tuning: clearer motion and a simpler
object fix most failures.

## 3. Present (about 2 minutes)

Open the finished run and press **Present** (or go to `/demo#<run id>`). Full screen it
(F11). The page shows the person and the robot side by side and highlights each robot
step as it happens.

Suggested script:

1. "Robots learn tasks from demonstrations, and demonstrations are expensive. We want a
   robot to learn from an ordinary phone video of a person."
2. Press **Play both**. "On the left, my teammate puts the block in the bowl. On the right,
   a Franka arm doing the same task in MuJoCo, with real physics: it has to actually grip
   and carry the block."
3. Point at the step list. "GPT-6 Astra watched the video and wrote this plan, one
   language action per step with a reason, the same format as the EgoLAP paper. Then the
   simulator checked it, and the model compared the result with the video and fixed it."
4. Click a step to jump the robot to it (the grasp is the best one to show).
5. Back in the workbench, open **Rounds** to show what the model changed between
   attempts, if it took more than one.

## Fallback

If the venue network fails, everything already processed still plays: the presenter view
only reads files from `runs/`. The bundled sample (**Sample: Hand puts cube in bowl**) is
a backup clip that is known to work.
