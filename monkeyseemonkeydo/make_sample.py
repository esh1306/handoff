"""Render sample clips of a stand-in 'person' (hand + forearm) doing a pick-and-place, filmed
from the side. Lets you try the pipeline without your own video.

    python make_sample.py          # samples/hand_puts_cube_in_bowl.mp4 (primitive shapes)
    python make_sample.py --toy    # samples/hand_puts_dinosaur_in_bowl.mp4 (scanned objects)
"""
import sys
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


def toy_xml():
    """XML with the scanned triceratops (carried) and turquoise bowl instead of cube and box."""
    objs = Path(__file__).parent / "assets" / "scanned_objects"
    dino, bowl = objs / "Great_Dinos_Triceratops_Toy", objs / "Room_Essentials_Bowl_Turquiose"
    assets = "".join(f'<texture name="{n}_t" type="2d" file="{d / "texture.png"}"/>'
                     f'<material name="{n}_m" texture="{n}_t"/><mesh name="{n}" file="{d / "model.obj"}"/>'
                     for n, d in (("dino", dino), ("bowl", bowl)))
    xml = XML.replace("<worldbody>", f"<asset>{assets}</asset><worldbody>")
    start, end = xml.index('<body name="bowl"'), xml.index("</body>", xml.index('<body name="bowl"')) + 7
    xml = xml[:start] + '<body name="bowl" pos=".05 .18 0"><geom type="mesh" mesh="bowl" material="bowl_m"/></body>' + xml[end:]
    xml = xml.replace('<geom type="box" size=".025 .025 .025" rgba=".85 .1 .1 1"/>',
                      '<geom type="mesh" mesh="dino" material="dino_m" pos="0 0 -.025"/>')
    return xml


def main(out=Path(__file__).parent / "samples" / "hand_puts_cube_in_bowl.mp4", fps=30, xml=XML):
    out.parent.mkdir(exist_ok=True)
    m = mujoco.MjModel.from_xml_string(xml)
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
    if "--toy" in sys.argv:
        main(Path(__file__).parent / "samples" / "hand_puts_dinosaur_in_bowl.mp4", xml=toy_xml())
    else:
        main()
