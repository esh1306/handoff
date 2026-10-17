"""Robot adapters: where each robot's MJCF lives and how to drive it."""
from dataclasses import dataclass, field
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
# Menagerie clone: monkeyseemonkeydo/assets/mujoco_menagerie, or the repo root's mujoco_menagerie.
MENAGERIE = next((p for p in (ASSETS / "mujoco_menagerie", Path(__file__).parent.parent / "mujoco_menagerie")
                  if (p / "franka_emika_panda").exists()), ASSETS / "mujoco_menagerie")


@dataclass
class Robot:
    name: str
    xml: Path
    tcp_body: str            # body the tool-centre-point site is attached to
    tcp_pos: tuple           # site offset inside that body (between the finger pads)
    arm_joints: list
    arm_actuators: list      # position-controlled, same order as arm_joints
    finger_bodies: tuple
    gripper_actuators: list
    gripper_open: list
    gripper_closed: list
    max_opening: float       # metres between the finger pads when fully open
    orient_ik: bool          # True: IK holds the gripper pointing down + yaw; False: position only
    home: dict = field(default_factory=dict)  # joint name -> initial qpos
    wrist_actuator: str | None = None         # position-only robots: yaw drives this joint
    wrist_cam: tuple | None = None            # (pos, look-at point, up hint) in tcp_body coordinates
    gripper_joints: tuple = ()                # finger joints, for the gripper opening in the state
    gripper_qpos_open: float = 0.0            # finger joint position when open / closed
    gripper_qpos_closed: float = 0.0
    notes: str = ""


ROBOTS = {
    "panda": Robot(
        name="panda",
        xml=MENAGERIE / "franka_emika_panda" / "panda.xml",
        tcp_body="hand",
        tcp_pos=(0, 0, 0.1034),
        arm_joints=[f"joint{i}" for i in range(1, 8)],
        arm_actuators=[f"actuator{i}" for i in range(1, 8)],
        finger_bodies=("left_finger", "right_finger"),
        gripper_actuators=["actuator8"],
        gripper_open=[255],
        gripper_closed=[0],
        max_opening=0.08,
        orient_ik=True,
        wrist_cam=((0.1, 0, -0.02), (0, 0, 0.13), (0, 0, -1)),  # beside the hand, aimed past the fingertips
        gripper_joints=("finger_joint1", "finger_joint2"),
        gripper_qpos_open=0.04,
        gripper_qpos_closed=0.0,
        home={"joint1": 0, "joint2": 0, "joint3": 0, "joint4": -1.57079, "joint5": 0,
              "joint6": 1.57079, "joint7": -0.7853, "finger_joint1": 0.04, "finger_joint2": 0.04},
        notes=("7-DOF Franka Emika Panda with a parallel gripper (max opening 8 cm). "
               "Its base sits at the origin on the table. Comfortable reach: 0.3-0.75 m "
               "horizontally from the base, tool height 0.0-0.6 m. The gripper always points "
               "straight down; you choose its yaw."),
    ),
    "qarm": Robot(
        name="qarm",
        xml=ASSETS / "qarm_robot.xml",
        tcp_body="wrist",
        tcp_pos=(0.125, 0, 0),
        arm_joints=["base_yaw", "shoulder", "elbow"],
        arm_actuators=["base_yaw", "shoulder", "elbow"],
        finger_bodies=("finger_l", "finger_r"),
        gripper_actuators=["finger_l", "finger_r"],
        gripper_open=[0, 0],
        gripper_closed=[0.03, 0.03],
        max_opening=0.07,
        orient_ik=False,
        wrist_actuator="wrist_roll",
        wrist_cam=((0.085, 0, 0.04), (0.2, 0, 0), (0, 0, 1)),  # front edge of the wrist, aimed between the fingers
        gripper_joints=("finger_l", "finger_r"),
        gripper_qpos_open=0.0,
        gripper_qpos_closed=0.03,
        notes=("4-DOF QArm-like arm (base yaw, shoulder pitch, elbow pitch, wrist roll) with a "
               "two-finger gripper (max opening 7 cm). Base at the origin; shoulder 0.14 m above "
               "the table, upper arm 0.35 m, forearm 0.25 m. Reach roughly 0.2-0.5 m from the base. "
               "Only the gripper POSITION is controlled: the gripper points roughly along the "
               "forearm, so it approaches objects from the side, not from above. Prefer objects "
               "raised on a support (z >= 0.08 m) and grasp heights near the shoulder. "
               "yaw sets the wrist roll (0 = fingers close horizontally)."),
    ),
}
