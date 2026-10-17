"""Prompts for the analyse -> simulate -> critique loop."""
import json

from scene import CATALOG


def catalog_text():
    """The scanned objects the model may use instead of primitive shapes."""
    lines = []
    for key, c in CATALOG.items():
        w, d, h = c["size_cm"]
        grasp = (f"; grasp at yaw {c['grasp_yaw']} deg, tool height {c['grasp_height']:.0%} of its height"
                 if c["kind"] == "graspable" else "")
        lines.append(f'- "{key}": {c["label"]}, {w} x {d} x {h} cm, {c["kind"]}{grasp}')
    return ("Realistic scanned objects (use one when it plausibly matches an object in the video, otherwise a "
            'primitive shape): {"name": ..., "shape": "mesh", "mesh": "<key>", "pos": [x, y, z], "yaw": deg}. '
            "Their size is fixed (omit size); pos is the object's centre, so on the table z = half its height.\n"
            + "\n".join(lines))

SCHEMA = """\
{
  "task_description": "one sentence: what the person accomplishes",
  "reasoning": "how you estimated object sizes and positions, and how you mapped the person's hand motion onto the robot",
  "camera": {"lookat": [x, y, z], "distance": metres, "azimuth": degrees, "elevation": degrees},
  "objects": [
    {"name": "red_cube", "shape": "box | cylinder | sphere | container | mesh",
     "size": [...], "pos": [x, y, z], "yaw": degrees, "rgba": [r, g, b, 1],
     "mass": kg, "fixed": false}
  ],
  "plan": [
    {"action": "move_to", "pos": [x, y, z], "yaw": degrees, "language": "move above the cube", "reason": "..."},
    {"action": "move_by", "delta_cm": [dx, dy, dz], "language": "move down 12 cm", "reason": "..."},
    {"action": "close", "language": "close gripper", "reason": "..."},
    {"action": "open", "language": "open gripper", "reason": "..."},
    {"action": "wait", "seconds": 0.5, "language": "pause", "reason": "..."}
  ],
  "success": [
    {"type": "object_on", "object": "red_cube", "support": "blue_bin"},
    {"type": "object_near", "object": "red_cube", "target": [x, y, z], "tolerance": 0.04},
    {"type": "object_lifted", "object": "red_cube", "min_height": 0.05}
  ],
  "done": false,
  "changes": "what you changed compared with the previous attempt, and why"
}"""

RULES = """\
World frame: metres, robot base at (0, 0, 0) on a table whose top is z = 0. +x points forward
from the robot, +y to the robot's left, +z up. The robot replaces the person: keep the objects'
layout relative to each other as in the video, but place everything inside the robot's reach.

Objects:
- box size = half-extents [hx, hy, hz]; cylinder size = [radius, half_height] (upright);
  sphere size = [radius]; container = open-top box (bin, bowl, cup, tray) with OUTER
  half-extents [hx, hy, hz]. Model concave things (bowls, mugs, boxes with openings) as containers.
- pos is the object's centre. Something resting on the table has z = its half height.
- Use "fixed": true only for things that never move (shelves, stands, heavy furniture).
- Estimate real sizes from familiar references (hands ~ 9 cm wide, cans ~ 6.6 cm across).

Plan (the robot's actions, retargeted from the person's hand):
- Positions are for the gripper's tool point, which sits between the finger pads.
- Each step gets a short "language" action (e.g. "move down 12 cm, close gripper") and a
  one-line physics "reason" (contact, geometry, clearance, why this motion comes next).
- Approach from above/behind with clearance, then descend; lift before carrying; lower
  before releasing. Grasp objects narrower than the gripper's max opening.
- move_by deltas are in centimetres; move_to positions are in metres.

Success: 1-3 checks that are true only when the task in the video has been reproduced.
Camera: choose it so the simulation is viewed from roughly the same angle as the video
(azimuth 180 looks along +x from behind the robot; elevation is negative when looking down)."""


def analyse_prompt(robot_notes, closing_axis, video_info, task_hint=None):
    hint = f"\nThe user says the task is: {task_hint}\n" if task_hint else ""
    return f"""You are turning a third-person video of a person doing a manipulation task into a MuJoCo
simulation in which a robot performs the same task.

Video: {video_info['duration_s']} s, {video_info['width']}x{video_info['height']}. The frames above are in time order.
{hint}
Robot: {robot_notes}
At yaw 0 the fingers close along world axis {closing_axis}; yaw rotates the gripper about +z.

{RULES}

{catalog_text()}

Reply with ONLY one JSON object in this format (no other text):
{SCHEMA}"""


def critique_prompt(spec, metrics, iteration, max_iters):
    return f"""Iteration {iteration} of {max_iters}. Above are frames from the original video, then frames
from the simulation of your previous answer (labelled with the plan step they follow).

Your previous answer:
{json.dumps(spec, indent=1)}

What happened in the simulation (measured, not estimated):
{json.dumps(metrics, indent=1)}

Compare the simulation with the video:
1. Does the scene match (objects, sizes, colours, layout)? Is the camera similar?
2. Did the robot do what the person did? Look at steps that were unreachable, had large
   tcp_error_mm, closed on nothing (empty "grasped"), dropped an object, or knocked things over.
3. Are the success checks a faithful test of the task?

Then reply with ONLY a complete revised JSON object in the same format (no other text).
Set "done": true only if the simulation reproduces the task and every success check passed.
Fix the actual cause of each failure (geometry, positions, approach height, yaw, ordering)
rather than loosening the success checks.

Rules, for reference:
{RULES}

{catalog_text()}"""
