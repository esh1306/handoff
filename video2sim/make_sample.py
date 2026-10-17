"""Render samples/hand_puts_cube_in_bowl.mp4: a stand-in 'person' (hand + forearm) moving
a cube into a bowl, filmed from the side. Lets you try the pipeline without your own video."""
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np

XML = """
<mujoco>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <light pos="1 -1.5 2" dir="-.4 .6 -.6" diffuse=".3 .3 .3"/>
    <geom type="plane" size="2 2 .1" rgba=".55 .42 .3 1"/>
    <body name="bowl" pos=".05 .18 .035">
      <geom type="box" pos="0 0 -.031" size=".07 .07 .004" rgba=".95 .95 .92 1"/>
      <geom type="box" pos=".066 0 0" size=".004 .07 .035" rgba=".95 .95 .92 1"/>
      <geom type="box" pos="-.066 0 0" size=".004 .07 .035" rgba=".95 .95 .92 1"/>
      <geom type="box" pos="0 .066 0" size=".07 .004 .035" rgba=".95 .95 .92 1"/>
      <geom type="box" pos="0 -.066 0" size=".07 .004 .035" rgba=".95 .95 .92 1"/>
    </body>
    <body name="cube" mocap="true" pos="0 -.15 .025">
      <geom type="box" size=".025 .025 .025" rgba=".85 .1 .1 1"/>
    </body>
    <body name="hand" mocap="true" pos="-.1 -.35 .3">
      <geom type="box" size=".045 .02 .05" pos="0 0 .02" rgba=".87 .67 .52 1"/>
      <geom type="capsule" fromto="0 0 .07 -.08 -.2 .35" size=".03" rgba=".2 .3 .55 1"/>
      <geom type="capsule" fromto=".04 0 -.03 .04 0 -.0" size=".009" rgba=".87 .67 .52 1"/>
      <geom type="capsule" fromto="-.04 0 -.03 -.04 0 -.0" size=".009" rgba=".87 .67 .52 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# (time s, hand position, holding cube?)
KEYS = [(0.0, [-0.1, -0.35, 0.30], False), (1.2, [0.0, -0.15, 0.16], False),
        (2.0, [0.0, -0.15, 0.065], False), (2.5, [0.0, -0.15, 0.065], True),
        (3.4, [0.02, 0.0, 0.22], True), (4.3, [0.05, 0.18, 0.14], True),
        (4.8, [0.05, 0.18, 0.09], True), (5.2, [0.05, 0.18, 0.09], False),
        (6.4, [-0.1, -0.05, 0.32], False)]


def hand_at(t):
    for (t0, p0, _), (t1, p1, h1) in zip(KEYS, KEYS[1:]):
        if t <= t1:
            a = (t - t0) / (t1 - t0)
            a = a * a * (3 - 2 * a)
            return np.array(p0) + (np.array(p1) - np.array(p0)) * a, h1
    return np.array(KEYS[-1][1]), False


def main(out=Path(__file__).parent / "samples" / "hand_puts_cube_in_bowl.mp4", fps=30):
    out.parent.mkdir(exist_ok=True)
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    hand, cube = m.body("hand").mocapid[0], m.body("cube").mocapid[0]
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [0.0, 0.0, 0.08], 1.0, 0, -22
    r = mujoco.Renderer(m, 720, 1280)
    w = imageio_ffmpeg.write_frames(str(out), (1280, 720), fps=fps, codec="libx264", quality=7)
    w.send(None)
    cube_pos = np.array([0.0, -0.15, 0.025])
    for i in range(int(7.0 * fps)):
        p, holding = hand_at(i / fps)
        d.mocap_pos[hand] = p
        if holding:
            cube_pos = p - [0, 0, 0.04]
        elif cube_pos[2] > 0.025 and np.linalg.norm(cube_pos[:2] - [0.05, 0.18]) < 0.05:
            cube_pos[2] = max(0.033 + 0.025 - 0.027, cube_pos[2] - 0.02)  # drop to the bowl floor
        d.mocap_pos[cube] = cube_pos
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        w.send(np.ascontiguousarray(r.render()))
    w.close()
    r.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
