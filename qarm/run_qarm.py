"""QArm-like arm with a slider control panel.

Two ways to move the tool tip (yellow dot):
  * Control panel sliders for X / Y / Z, wrist roll and gripper.
  * In the 3D viewer: double-click the green target ball, then hold Ctrl and
    right-drag to move it (Ctrl + left-drag rotates nothing here).
Inverse kinematics (damped least squares) solves base/shoulder/elbow so the
tip follows the target.
"""
import sys, time
import tkinter as tk
import numpy as np
import mujoco, mujoco.viewer

m = mujoco.MjModel.from_xml_path(sys.argv[1])
d = mujoco.MjData(m)
ik = mujoco.MjData(m)  # scratch data for IK, never stepped

TIP = m.site("tip").id
TARGET = m.body("target").mocapid[0]
NAMES = ("base_yaw", "shoulder", "elbow")
ARM = [m.joint(n).qposadr[0] for n in NAMES]
ARM_DOF = [m.joint(n).dofadr[0] for n in NAMES]
LO, HI = m.jnt_range[[m.joint(n).id for n in NAMES]].T
A_WRIST, A_FL, A_FR = (m.actuator(n).id for n in ("wrist_roll", "finger_l", "finger_r"))


def solve_ik(target, q0):
    """Return arm joint angles putting the tip at `target`, and the residual."""
    ik.qpos[:] = d.qpos
    ik.qpos[ARM] = q0
    jac = np.zeros((3, m.nv))
    for _ in range(200):
        mujoco.mj_kinematics(m, ik)
        mujoco.mj_comPos(m, ik)  # needed by mj_jacSite
        err = target - ik.site_xpos[TIP]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(m, ik, jac, None, TIP)
        J = jac[:, ARM_DOF]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
        ik.qpos[ARM] = np.clip(ik.qpos[ARM] + dq, LO, HI)
    return ik.qpos[ARM].copy(), np.linalg.norm(err)


def home_tip():
    tmp = mujoco.MjData(m)  # all-zero pose
    mujoco.mj_kinematics(m, tmp)
    return tmp.site_xpos[TIP].copy()


# ---------------- control panel ----------------
root = tk.Tk()
root.title("QArm control")
root.attributes("-topmost", True)
sliders = {}


def add_slider(label, lo, hi, res, init, row):
    tk.Label(root, text=label, width=14, anchor="w").grid(row=row, column=0, padx=6)
    s = tk.Scale(root, from_=lo, to=hi, resolution=res, orient="horizontal",
                 length=320, command=lambda _v: apply_sliders())
    s.set(init)
    s.grid(row=row, column=1, padx=6)
    sliders[label] = s


def apply_sliders():
    d.mocap_pos[TARGET] = [sliders["X (fwd) m"].get(),
                           sliders["Y (left) m"].get(),
                           sliders["Z (up) m"].get()]
    d.ctrl[A_WRIST] = sliders["Wrist roll rad"].get()
    d.ctrl[[A_FL, A_FR]] = 0.03 * sliders["Gripper close %"].get() / 100


def sync_sliders_to_target():
    """Keep sliders in step when the target is dragged in the viewer."""
    x, y, z = d.mocap_pos[TARGET]
    for name, val in (("X (fwd) m", x), ("Y (left) m", y), ("Z (up) m", z)):
        s = sliders[name]
        if abs(s.get() - val) > 0.003:
            s.configure(command=lambda _v: None)
            s.set(round(val, 3))
            s.configure(command=lambda _v: apply_sliders())


hx, hy, hz = home_tip()
add_slider("X (fwd) m", -0.55, 0.60, 0.005, hx, 0)
add_slider("Y (left) m", -0.60, 0.60, 0.005, hy, 1)
add_slider("Z (up) m", 0.00, 0.85, 0.005, hz, 2)
add_slider("Wrist roll rad", -2.79, 2.79, 0.05, 0, 3)
add_slider("Gripper close %", 0, 100, 1, 0, 4)


def go(x, y, z):
    sliders["X (fwd) m"].set(x); sliders["Y (left) m"].set(y); sliders["Z (up) m"].set(z)
    apply_sliders()


btns = tk.Frame(root)
btns.grid(row=5, column=0, columnspan=2, pady=6)
tk.Button(btns, text="Home", width=12, command=lambda: go(hx, hy, hz)).pack(side="left", padx=4)
tk.Button(btns, text="Above cube", width=12, command=lambda: go(0.375, 0, 0.25)).pack(side="left", padx=4)
status = tk.Label(root, text="", font=("Consolas", 10), anchor="w", justify="left")
status.grid(row=6, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6))

apply_sliders()

# ---------------- simulation loop ----------------
viewer = mujoco.viewer.launch_passive(m, d)
q_cmd = np.zeros(3)
frame = [0]


def tick():
    global q_cmd
    if not viewer.is_running():
        root.destroy()
        return
    start = time.time()
    with viewer.lock():
        q_cmd, ik_err = solve_ik(d.mocap_pos[TARGET].copy(), q_cmd)
        d.ctrl[:3] = q_cmd
        for _ in range(8):  # 8 x 2 ms = 16 ms of sim per frame
            mujoco.mj_step(m, d)
    viewer.sync()
    frame[0] += 1
    if frame[0] % 6 == 0:
        sync_sliders_to_target()
        tip, tgt = d.site_xpos[TIP], d.mocap_pos[TARGET]
        note = "  TARGET OUT OF REACH" if ik_err > 5e-3 else ""
        status.configure(text=f"tip    {np.round(tip, 3)}\n"
                              f"error  {1000 * np.linalg.norm(tgt - tip):5.1f} mm{note}")
    root.after(max(1, int(16 - 1000 * (time.time() - start))), tick)


root.protocol("WM_DELETE_WINDOW", lambda: (viewer.close(), root.destroy()))
root.after(16, tick)
root.mainloop()
