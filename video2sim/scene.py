"""Turn a scene spec (JSON from the model) into a compiled MuJoCo model."""
import math
import mujoco
import numpy as np

from robots import Robot

SHAPES = {"box", "cylinder", "sphere", "container"}
ACTIONS = {"move_to", "move_by", "open", "close", "wait"}
CHECKS = {"object_on", "object_near", "object_lifted"}
SIZE_LEN = {"box": 3, "cylinder": 2, "sphere": 1, "container": 3}


def half_height(obj):
    s = obj["size"]
    return s[{"box": 2, "container": 2, "cylinder": 1, "sphere": 0}[obj["shape"]]]


def validate(spec):
    """Return a list of problems with a spec; empty means OK."""
    errs = []
    for key in ("objects", "plan", "success"):
        if not isinstance(spec.get(key), list):
            errs.append(f"'{key}' must be a list")
    if errs:
        return errs
    names = set()
    for i, o in enumerate(spec["objects"]):
        n = o.get("name")
        if not n or n in names:
            errs.append(f"objects[{i}]: missing or duplicate name {n!r}")
        names.add(n)
        if o.get("shape") not in SHAPES:
            errs.append(f"objects[{i}] {n}: shape must be one of {sorted(SHAPES)}")
            continue
        size = o.get("size")
        if (not isinstance(size, list) or len(size) != SIZE_LEN[o["shape"]]
                or any(not isinstance(v, (int, float)) or v <= 0 for v in size)):
            errs.append(f"objects[{i}] {n}: size must be {SIZE_LEN[o['shape']]} positive numbers")
        if not (isinstance(o.get("pos"), list) and len(o["pos"]) == 3):
            errs.append(f"objects[{i}] {n}: pos must be [x, y, z]")
    for i, step in enumerate(spec["plan"]):
        a = step.get("action")
        if a not in ACTIONS:
            errs.append(f"plan[{i}]: action must be one of {sorted(ACTIONS)}")
        elif a == "move_to" and not (isinstance(step.get("pos"), list) and len(step["pos"]) == 3):
            errs.append(f"plan[{i}]: move_to needs pos [x, y, z]")
        elif a == "move_by" and not (isinstance(step.get("delta_cm"), list) and len(step["delta_cm"]) == 3):
            errs.append(f"plan[{i}]: move_by needs delta_cm [dx, dy, dz]")
    for i, c in enumerate(spec["success"]):
        if c.get("type") not in CHECKS:
            errs.append(f"success[{i}]: type must be one of {sorted(CHECKS)}")
        elif c.get("object") not in names:
            errs.append(f"success[{i}]: unknown object {c.get('object')!r}")
        elif c["type"] == "object_on" and c.get("support") not in names:
            errs.append(f"success[{i}]: unknown support {c.get('support')!r}")
    if not spec["plan"]:
        errs.append("plan is empty")
    return errs


def _add_object(world, o):
    yaw = math.radians(o.get("yaw", 0))
    pos = list(map(float, o["pos"]))
    pos[2] = max(pos[2], half_height(o))  # never start below the table
    body = world.add_body(name=o["name"], pos=pos, quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    if not o.get("fixed", False):
        body.add_freejoint()
    rgba = o.get("rgba", [0.6, 0.6, 0.6, 1])
    common = dict(rgba=rgba, condim=4, friction=[o.get("friction", 1.2), 0.005, 0.0001])
    s, shape = o["size"], o["shape"]

    if shape == "container":  # open-top box: floor + four walls
        hx, hy, hz = s
        t = min(o.get("wall", 0.006), hx / 2, hy / 2, hz)
        parts = [([0, 0, -hz + t / 2], [hx, hy, t / 2]),
                 ([hx - t / 2, 0, 0], [t / 2, hy, hz]), ([-hx + t / 2, 0, 0], [t / 2, hy, hz]),
                 ([0, hy - t / 2, 0], [hx, t / 2, hz]), ([0, -hy + t / 2, 0], [hx, t / 2, hz])]
        mass = o.get("mass", 0.2) / len(parts)
        for p, hs in parts:
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=p, size=hs, mass=mass, **common)
        return

    gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
             "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}[shape]
    size = list(s) + [0] * (3 - len(s))
    body.add_geom(type=gtype, size=size, mass=o.get("mass", 0.05), **common)


def _look_at(pos, target, up):
    """Quaternion for a MuJoCo camera at pos looking at target (cameras look along -z, +y is image up)."""
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).flatten())
    return q.tolist()


def build(spec, robot: Robot):
    """Return (MjModel, MjSpec) for the robot plus the spec's objects on a table at z=0."""
    ms = mujoco.MjSpec.from_file(str(robot.xml))
    for key in list(ms.keys):  # robot keyframes no longer match qpos once objects are added
        ms.delete(key)
    ms.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    ms.visual.global_.offwidth = 1280
    ms.visual.global_.offheight = 960
    ms.body(robot.tcp_body).add_site(name="tcp", pos=list(robot.tcp_pos), size=[0.006] * 3,
                                     rgba=[1, 1, 0, 1], group=4)
    if robot.wrist_cam:
        pos, target, up = robot.wrist_cam
        ms.body(robot.tcp_body).add_camera(name="wrist", pos=list(pos), quat=_look_at(pos, target, up), fovy=70)

    world = ms.worldbody
    world.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
    world.add_light(pos=[1.5, -1.5, 2], dir=[-0.5, 0.5, -0.6], diffuse=[0.3, 0.3, 0.3])
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                   rgba=[0.8, 0.78, 0.74, 1])
    for o in spec["objects"]:
        _add_object(world, o)
    return ms.compile(), ms
