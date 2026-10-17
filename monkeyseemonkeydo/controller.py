"""Execute a plan of primitives on a robot in MuJoCo, record video and measure the outcome."""
import math
import imageio_ffmpeg
import numpy as np
import mujoco

from robots import Robot
from scene import half_height, holds_inside

DEFAULT_CAMERA = {"lookat": [0.45, 0.0, 0.1], "distance": 1.6, "azimuth": 150, "elevation": -25}


def _rotz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Sim:
    CTRL_DT = 0.02  # control period (s)

    def __init__(self, model, robot: Robot, spec, video_path=None, size=(480, 640), fps=30, on_tick=None,
                 camera_jitter=None):
        self.m, self.robot, self.spec = model, robot, spec
        self.d = mujoco.MjData(model)
        self.scratch = mujoco.MjData(model)  # IK workspace, never stepped
        m = model
        self.site = m.site("tcp").id
        self.qadr = np.array([m.joint(j).qposadr[0] for j in robot.arm_joints])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in robot.arm_joints])
        jid = [m.joint(j).id for j in robot.arm_joints]
        limited = m.jnt_limited[jid].astype(bool)
        self.lo = np.where(limited, m.jnt_range[jid, 0], -np.inf)
        self.hi = np.where(limited, m.jnt_range[jid, 1], np.inf)
        self.arm_act = [m.actuator(a).id for a in robot.arm_actuators]
        self.grip_act = [m.actuator(a).id for a in robot.gripper_actuators]
        self.wrist_act = m.actuator(robot.wrist_actuator).id if robot.wrist_actuator else None
        self.fingers = {m.body(b).id for b in robot.finger_bodies}
        self.objects = {o["name"]: m.body(o["name"]).id for o in spec["objects"]}
        self.movable = {o["name"] for o in spec["objects"] if not o.get("fixed", False)}
        self.on_tick = on_tick

        self.record = video_path is not None
        self.fps = fps
        self.writer = None
        self.mirrored = False
        self.renderer = None
        self.recorder = None   # optional callable(sim), run every control tick (dataset export)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam = {**DEFAULT_CAMERA, **(spec.get("camera") or {})}
        az, el = cam["azimuth"], cam["elevation"]
        if math.cos(math.radians(az)) > 0.3:
            # Camera behind the robot (looking along +x): its body would hide the task,
            # so film it from the front instead. Left and right appear swapped.
            az = 180 - az
            self.mirrored = True
        if camera_jitter:
            az, el = az + camera_jitter[0], el + camera_jitter[1]
        self.cam.lookat[:] = cam["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = max(cam["distance"], 1.2), az, el
        if self.record:
            # H.264 so browsers can play it
            self.writer = imageio_ffmpeg.write_frames(str(video_path), (size[1], size[0]), fps=fps,
                                                      codec="libx264", quality=7)
            self.writer.send(None)
            self.renderer = mujoco.Renderer(m, *size)
        self.keyframes = []   # (label, RGB) at the end of each plan step
        self.reset()

    # ---------- setup ----------
    def reset(self):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        for j, v in self.robot.home.items():
            d.qpos[m.joint(j).qposadr[0]] = v
        self.q_cmd = d.qpos[self.qadr].copy()
        d.ctrl[self.arm_act] = self.q_cmd
        d.ctrl[self.grip_act] = self.robot.gripper_open
        mujoco.mj_forward(m, d)
        self.home_rot = d.site_xmat[self.site].reshape(3, 3).copy()
        self.target = d.site_xpos[self.site].copy()
        self.yaw = 0.0
        self.grip_closed = False
        self.step_index = -1
        self.next_frame = 0.0
        self.max_z = {n: d.xpos[b][2] for n, b in self.objects.items()}
        self.initial = {n: d.xpos[b].copy() for n, b in self.objects.items()}
        self.keyframes.clear()

    def closing_axis(self):
        """World direction the fingers close along at yaw=0 (for the prompt)."""
        if self.robot.orient_ik:
            return self.home_rot[:, 1].round(2).tolist()
        return [0.0, 1.0, 0.0]

    # ---------- low level ----------
    def ik(self, pos, yaw):
        """Damped least squares; returns (joint targets, position residual in m)."""
        m, s = self.m, self.scratch
        s.qpos[:] = self.d.qpos
        s.qpos[self.qadr] = self.q_cmd
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        if self.robot.orient_ik:
            target_quat = np.zeros(4)
            mujoco.mju_mat2Quat(target_quat, (_rotz(yaw) @ self.home_rot).flatten())
        cur, neg, errq, errr = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
        for _ in range(60):
            mujoco.mj_kinematics(m, s)
            mujoco.mj_comPos(m, s)  # needed by mj_jacSite
            errp = pos - s.site_xpos[self.site]
            mujoco.mj_jacSite(m, s, jacp, jacr, self.site)
            if self.robot.orient_ik:
                mujoco.mju_mat2Quat(cur, s.site_xmat[self.site])
                mujoco.mju_negQuat(neg, cur)
                mujoco.mju_mulQuat(errq, target_quat, neg)
                mujoco.mju_quat2Vel(errr, errq, 1.0)
                err = np.concatenate([errp, 0.5 * errr])
                J = np.vstack([jacp[:, self.dadr], 0.5 * jacr[:, self.dadr]])
            else:
                err, J = errp, jacp[:, self.dadr]
            if np.linalg.norm(err) < 1e-4:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(len(err)), err)
            s.qpos[self.qadr] = np.clip(s.qpos[self.qadr] + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        mujoco.mj_kinematics(m, s)
        return s.qpos[self.qadr].copy(), float(np.linalg.norm(pos - s.site_xpos[self.site]))

    def tick(self):
        m, d = self.m, self.d
        for _ in range(max(1, round(self.CTRL_DT / m.opt.timestep))):
            mujoco.mj_step(m, d)
        for n, b in self.objects.items():
            self.max_z[n] = max(self.max_z[n], d.xpos[b][2])
        if self.record and d.time >= self.next_frame:
            self.writer.send(np.ascontiguousarray(self.render()))
            self.next_frame += 1.0 / self.fps
        if self.recorder:
            self.recorder(self)
        if self.on_tick:
            self.on_tick()

    def close(self):
        """Free the GL context now; a stale one left to the garbage collector blacks out later renders."""
        if self.writer:
            self.writer.close()
            self.writer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def render(self, camera=None):
        """RGB image from the scene camera, or from a named model camera (e.g. "wrist")."""
        self.renderer.update_scene(self.d, camera=self.cam if camera is None else camera)
        return self.renderer.render().copy()

    def hold(self, seconds):
        for _ in range(max(1, int(seconds / self.CTRL_DT))):
            self.tick()

    def grasped(self):
        """Objects touched by both fingers right now."""
        m, d = self.m, self.d
        touch = {}
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for f, o in ((b1, b2), (b2, b1)):
                if f in self.fingers:
                    for n, b in self.objects.items():
                        if b == o:
                            touch.setdefault(n, set()).add(f)
        return sorted(n for n, fs in touch.items() if len(fs) == 2)

    # ---------- primitives ----------
    def move_to(self, pos, yaw=None, speed=0.25):
        start, end = self.target.copy(), np.asarray(pos, float)
        y0 = self.yaw
        y1 = y0 if yaw is None else float(yaw)
        n = max(1, int(max(np.linalg.norm(end - start) / speed, abs(y1 - y0) / 90, 0.3) / self.CTRL_DT))
        worst = 0.0
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)  # smoothstep
            p, y = start + (end - start) * a, y0 + (y1 - y0) * a
            self.q_cmd, res = self.ik(p, y)
            worst = max(worst, res)
            self.d.ctrl[self.arm_act] = self.q_cmd
            if self.wrist_act is not None:
                self.d.ctrl[self.wrist_act] = math.radians(y)
            self.tick()
        self.target, self.yaw = end, y1
        self.hold(0.3)
        return worst

    def gripper(self, close):
        self.grip_closed = bool(close)
        self.d.ctrl[self.grip_act] = self.robot.gripper_closed if close else self.robot.gripper_open
        self.hold(0.8)

    # ---------- plan ----------
    def run(self, plan):
        steps = []
        if self.record:
            self.keyframes.append(("sim initial state", self.render()))
        for i, step in enumerate(plan):
            self.step_index = i
            a = step["action"]
            # sim time == time in sim.mp4, so the UI can follow the steps while the video plays
            info = {"i": i, "action": a, "language": step.get("language", ""), "t_start": round(self.d.time, 2)}
            if a in ("move_to", "move_by"):
                goal = (step["pos"] if a == "move_to"
                        else self.target + np.asarray(step["delta_cm"], float) / 100)
                res = self.move_to(goal, step.get("yaw"), step.get("speed", 0.25))
                tcp = self.d.site_xpos[self.site]
                info.update(goal=np.round(goal, 3).tolist(), tcp=np.round(tcp, 3).tolist(),
                            tcp_error_mm=round(1000 * float(np.linalg.norm(tcp - goal)), 1),
                            reachable=res < 0.005)
            elif a in ("open", "close"):
                self.gripper(a == "close")
            elif a == "wait":
                self.hold(float(step.get("seconds", 0.5)))
            info["grasped"] = self.grasped()
            info["t_end"] = round(self.d.time, 2)
            steps.append(info)
            if self.record:
                self.keyframes.append((f"sim after step {i}: {a} {step.get('language', '')}".strip(),
                                       self.render()))
        self.hold(1.0)  # let things settle before judging
        if self.writer:
            self.writer.close()
            self.writer = None
        return self.metrics(steps)

    def metrics(self, steps):
        d = self.d
        final = {n: d.xpos[b].copy() for n, b in self.objects.items()}
        objs = {n: {"initial": np.round(self.initial[n], 3).tolist(),
                    "final": np.round(final[n], 3).tolist(),
                    "max_lift_cm": round(100 * (self.max_z[n] - self.initial[n][2]), 1)}
                for n in self.objects}
        checks = [self.check(c, final) for c in self.spec["success"]]
        out = {"steps": steps, "objects": objs, "success": checks,
               "all_success": bool(checks) and all(c["passed"] for c in checks),
               "sim_seconds": round(d.time, 2)}
        if self.mirrored:
            out["camera_note"] = ("Your camera would have been behind the robot, so the simulation frames "
                                  "are filmed from the front: left and right are swapped compared with the "
                                  "video. Judge the layout from the object positions, not the image.")
        return out

    def check(self, c, final):
        o = c["object"]
        p = final[o]
        if c["type"] == "object_near":
            dist = float(np.linalg.norm(p - np.asarray(c["target"], float)))
            tol = c.get("tolerance", 0.04)
            return {**c, "passed": bool(dist <= tol), "detail": f"{100 * dist:.1f} cm from target (tol {100 * tol:.0f} cm)"}
        if c["type"] == "object_lifted":
            lift = self.max_z[o] - self.initial[o][2]
            need = c.get("min_height", 0.05)
            return {**c, "passed": bool(lift >= need), "detail": f"max lift {100 * lift:.1f} cm (need {100 * need:.0f} cm)"}
        # object_on: object's centre inside the support's footprint and resting on/in it
        sup = next(s for s in self.spec["objects"] if s["name"] == c["support"])
        sp = final[c["support"]]
        R = self.d.xmat[self.objects[c["support"]]].reshape(3, 3)
        local = R.T @ (p - sp)
        sz = sup["size"]
        if sup["shape"] in ("box", "container", "mesh"):
            inside = abs(local[0]) <= sz[0] and abs(local[1]) <= sz[1]
        else:  # cylinder / sphere
            inside = math.hypot(local[0], local[1]) <= sz[0]
        hh = half_height(sup)
        if holds_inside(sup):
            level = -hh <= local[2] <= hh + 0.05
        else:
            level = local[2] >= hh - 0.01
        ok = bool(inside and level)
        return {**c, "passed": ok,
                "detail": f"offset from {c['support']} centre {np.round(100 * local, 1).tolist()} cm"}
