"""
VR Demo Collection for A2A Flow Matching

Collects expert demonstrations using Meta Quest 3 hand tracking
to teleoperate a Franka arm in the A2A MuJoCo simulation.

Usage:
    cd /home/choon/A2A/A2A_Flow_Matching
    export PYTHONPATH=$(pwd):$PYTHONPATH
    export PYTHONPATH=/home/choon/A2A/vr_teleop_control:$PYTHONPATH
    export PYTHONPATH=/home/choon/intuitive_autonomy/ia_teleopVR/pyroki/examples:$PYTHONPATH
    export DISPLAY=:1

    python roboverse_learn/il/vr_collect_demo.py \
        --task close_box \
        --num_demos 25 \
        --quest_ip 10.0.0.6 \
        --save_dir ./roboverse_demo/demo_mujoco/close_box-vr/robot-franka/success

Pipeline:
    Quest 3 right hand pose
        -> PyRoki IK (panda_arm_hand.urdf)
        -> Franka joint targets (7 DOF)
        -> A2A MuJoCo env.step()
        -> Record metadata.json + rgb.mp4
        -> data2zarr_dp.py -> .zarr -> train
"""

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# PyRoki
import pyroki as pk
import jax.numpy as jnp

# A2A / RoboVerse
from metasim.task.registry import get_task_class
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.utils.setup_util import get_robot
from metasim.utils.demo_util import get_traj

# Quest3 controller (from vr_teleop_control)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../vr_teleop_control'))
from utils.quest3_controller import Quest3Controller

FRANKA_URDF = os.path.join(
    os.path.dirname(__file__),
    '../../../RoboVerse/.venv/lib/python3.11/site-packages/'
    'isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/'
    'robots/franka_description/robots/panda_arm_hand.urdf'
)
FRANKA_EE_LINK = "panda_hand"
FRANKA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]
GRIPPER_OPEN  = 0.04   # metres per finger
GRIPPER_CLOSED = 0.0


def solve_franka_ik(robot, target_pos, target_wxyz, current_cfg=None):
    """Solve IK for Franka panda_hand EE target.

    Args:
        robot: pk.Robot loaded from panda_arm_hand.urdf
        target_pos: (3,) target position in world frame
        target_wxyz: (4,) target orientation as wxyz quaternion
        current_cfg: (N,) warm-start configuration (unused by solver but kept for API compat)

    Returns:
        cfg: (N,) joint angles in radians
    """
    target_link_index = list(robot.links.names).index(FRANKA_EE_LINK)

    import jax_dataclasses as jdc
    import jaxlie
    import jaxls

    @jdc.jit
    def _solve(robot, target_wxyz, target_pos):
        joint_var = robot.joint_var_cls(0)
        costs = [
            pk.costs.pose_cost_analytic_jac(
                robot, joint_var,
                jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(target_wxyz), target_pos
                ),
                jnp.array(target_link_index),
                pos_weight=50.0,
                ori_weight=5.0,
            ),
            pk.costs.limit_constraint(robot, joint_var),
        ]
        sol = (
            jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
            .analyze()
            .solve(verbose=False, linear_solver="dense_cholesky")
        )
        return sol[joint_var]

    cfg = np.array(_solve(robot, jnp.array(target_wxyz), jnp.array(target_pos)))
    return cfg


class DemoRecorder:
    """Records one demonstration episode in A2A training format."""

    def __init__(self, save_dir: str, demo_idx: int, fps: int = 30):
        self.save_dir = Path(save_dir) / f"demo_{demo_idx:04d}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.data = {
            "joint_qpos": [],
            "joint_qpos_target": [],
            "robot_root_state": [],
            "robot_ee_state": [],
            "robot_ee_state_target": [],
        }
        self.frames = []

    def record(self, joint_qpos, joint_qpos_target, rgb_frame,
               robot_root_state=None, robot_ee_state=None, robot_ee_state_target=None):
        """Record one timestep."""
        self.data["joint_qpos"].append(
            joint_qpos.tolist() if hasattr(joint_qpos, 'tolist') else list(joint_qpos)
        )
        self.data["joint_qpos_target"].append(
            joint_qpos_target.tolist() if hasattr(joint_qpos_target, 'tolist') else list(joint_qpos_target)
        )
        # Fallback zero states if EE states not available
        zeros7 = [0.0] * 7
        self.data["robot_root_state"].append(
            robot_root_state.tolist() if robot_root_state is not None else zeros7
        )
        self.data["robot_ee_state"].append(
            robot_ee_state.tolist() if robot_ee_state is not None else zeros7
        )
        self.data["robot_ee_state_target"].append(
            robot_ee_state_target.tolist() if robot_ee_state_target is not None else zeros7
        )
        # RGB frame: ensure HWC uint8
        if isinstance(rgb_frame, torch.Tensor):
            rgb_frame = rgb_frame.cpu().numpy()
        if rgb_frame.dtype != np.uint8:
            rgb_frame = (rgb_frame * 255).astype(np.uint8)
        self.frames.append(rgb_frame)

    def save(self):
        """Write metadata.json and rgb.mp4."""
        # metadata.json
        with open(self.save_dir / "metadata.json", "w") as f:
            json.dump(self.data, f)

        # rgb.mp4
        if self.frames:
            h, w = self.frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(self.save_dir / "rgb.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (w, h),
            )
            for frame in self.frames:
                # cv2 expects BGR
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()

        steps = len(self.frames)
        print(f"  Saved demo to {self.save_dir}  ({steps} steps)")
        return steps


def build_env(task_name: str, headless: bool = True):
    """Initialise the A2A MuJoCo task environment."""
    task_cls = get_task_class(task_name)

    camera = PinholeCameraCfg(
        name="camera0",
        data_types=["rgb"],
        width=256,
        height=256,
        pos=(1.5, 0.0, 1.5),
        look_at=(0.0, 0.0, 0.0),
    )

    scenario = task_cls.scenario.update(
        simulator="mujoco",
        num_envs=1,
        headless=headless,
        cameras=[camera],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)
    return env, task_cls


def build_action(joint_targets: np.ndarray, gripper: float):
    """Convert 7-DOF joint targets + gripper width to A2A action dict."""
    dof_targets = {}
    for i, name in enumerate(FRANKA_JOINT_NAMES):
        dof_targets[name] = float(joint_targets[i])
    dof_targets["panda_finger_joint1"] = gripper
    dof_targets["panda_finger_joint2"] = gripper
    return [{"franka": {"dof_pos_target": dof_targets}}]


def collect_demos(args):
    os.environ.setdefault("DISPLAY", ":1")
    os.environ.setdefault("MUJOCO_GL", "glx")

    # ── Environment ──────────────────────────────────────────────────────────
    print(f"Loading {args.task} in MuJoCo...")
    env, task_cls = build_env(args.task, headless=args.headless)
    robot_obj = get_robot("franka")

    # Load initial states from trajectory file for resets
    init_states = None
    try:
        traj_path = env.traj_filepath
        if traj_path and os.path.exists(traj_path):
            init_states, _, _ = get_traj(traj_path, robot_obj, env.handler)
            print(f"  Loaded {len(init_states)} initial states from trajectory")
    except Exception as e:
        print(f"  No trajectory file, using default reset: {e}")

    # ── PyRoki Franka robot ───────────────────────────────────────────────────
    print(f"Loading Franka URDF from {FRANKA_URDF}...")
    assert os.path.exists(FRANKA_URDF), f"Franka URDF not found: {FRANKA_URDF}"
    franka = pk.Robot.from_urdf_path(FRANKA_URDF)
    print(f"  Franka: {franka.joints.num_actuated_joints} actuated joints")
    # Warm up JIT
    print("  Warming up PyRoki JIT...")
    _ = solve_franka_ik(franka, np.array([0.4, 0.0, 0.5]),
                        np.array([1.0, 0.0, 0.0, 0.0]))
    print("  JIT ready.")

    # ── Quest 3 ───────────────────────────────────────────────────────────────
    quest = Quest3Controller(
        ip_address=args.quest_ip,
        vr_height=args.vr_height,
        rotation_offset_euler=args.rotation_offset,
        position_offset=args.position_offset,
        position_signs=args.position_signs,
        position_permutation=args.position_permutation,
    )
    quest_ok = quest.start(use_unity=args.use_unity)
    if not quest_ok:
        print("Quest 3 not connected — running in keyboard/debug mode")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ── Demo collection loop ──────────────────────────────────────────────────
    demo_idx = 0
    saved_demos = 0

    print(f"\nCollecting {args.num_demos} demos for task '{args.task}'")
    print("Controls:")
    print("  Right grip  = close gripper / activate arm")
    print("  Left grip   = save & reset episode")
    print("  Both grips  = discard & reset episode")
    print("-" * 50)

    while saved_demos < args.num_demos:
        demo_idx += 1
        print(f"\n[Demo {demo_idx}] Resetting environment...")

        # Reset
        reset_state = None
        if init_states:
            idx = (demo_idx - 1) % len(init_states)
            reset_state = [init_states[idx]]
        obs, _ = env.reset(states=reset_state)
        time.sleep(0.2)

        recorder = DemoRecorder(args.save_dir, saved_demos)
        current_cfg = np.zeros(franka.joints.num_actuated_joints)
        step = 0
        success_once = False
        print(f"  Running (max {args.max_steps} steps)...")

        while step < args.max_steps:
            # ── Observations ─────────────────────────────────────────────
            rgb = obs.cameras["camera0"].rgb[0]      # (H, W, 3) uint8
            joint_qpos = obs.robots["franka"].joint_pos[0].cpu().numpy()  # (9,)

            # ── Quest 3 input ─────────────────────────────────────────────
            if quest_ok:
                right_pos  = quest.right_pos.copy()
                right_quat = quest.right_quat.copy()  # wxyz
                right_grip = quest.right_grip
                left_grip  = quest.left_grip

                # Both grips = discard episode
                if right_grip and left_grip:
                    print("  Both grips: discarding episode.")
                    break

                # Left grip alone = save episode
                if left_grip and not right_grip:
                    print("  Left grip: saving episode.")
                    steps = recorder.save()
                    saved_demos += 1
                    print(f"  [{saved_demos}/{args.num_demos}] Saved ({steps} steps)")
                    break

                # Solve IK from Quest 3 right hand pose
                cfg = solve_franka_ik(franka, right_pos, right_quat, current_cfg)
                current_cfg = cfg
                gripper_val = GRIPPER_CLOSED if right_grip else GRIPPER_OPEN
            else:
                # No Quest 3: replay initial config (for testing pipeline)
                cfg = current_cfg
                gripper_val = GRIPPER_OPEN

            # ── Build and execute action ───────────────────────────────────
            arm_targets = cfg[:7]
            full_targets = np.concatenate([arm_targets, [gripper_val, gripper_val]])
            action = build_action(arm_targets, gripper_val)

            obs, _, success, timeout, _ = env.step(action)

            if success[0]:
                success_once = True

            # ── Record ────────────────────────────────────────────────────
            recorder.record(
                joint_qpos=joint_qpos,
                joint_qpos_target=full_targets,
                rgb_frame=rgb,
            )

            step += 1

            # Auto-save on success
            if success_once:
                print(f"  Task succeeded at step {step}!")
                steps = recorder.save()
                saved_demos += 1
                print(f"  [{saved_demos}/{args.num_demos}] Saved ({steps} steps)")
                break

            # Timeout without Quest (test mode)
            if not quest_ok and step >= args.max_steps:
                print(f"  Timeout at step {step} (no Quest 3, test mode)")
                steps = recorder.save()
                saved_demos += 1
                break

        if timeout and not success_once:
            print(f"  Episode timed out at step {step} (not saved)")

    env.close()
    print(f"\nDone! Collected {saved_demos} demos -> {args.save_dir}")
    print("\nNext step — convert to ZARR:")
    print(f"  python roboverse_learn/il/data2zarr_dp.py \\")
    print(f"    --task_name close_boxFrankaL0_obs:joint_pos_act:joint_pos \\")
    print(f"    --expert_data_num {saved_demos} \\")
    print(f"    --metadata_dir {args.save_dir} \\")
    print(f"    --observation_space joint_pos \\")
    print(f"    --action_space joint_pos")


def main():
    parser = argparse.ArgumentParser(description="VR Demo Collection for A2A")

    # Task
    parser.add_argument("--task", default="close_box",
                        help="Task name (e.g. close_box, pick_cube)")
    parser.add_argument("--num_demos", type=int, default=25,
                        help="Number of demos to collect")
    parser.add_argument("--max_steps", type=int, default=300,
                        help="Max steps per episode")
    parser.add_argument("--save_dir", default="./roboverse_demo/demo_mujoco/close_box-vr/robot-franka/success",
                        help="Directory to save demonstrations")
    parser.add_argument("--headless", action="store_true",
                        help="Run MuJoCo headless (no window)")

    # Quest 3
    parser.add_argument("--quest_ip", default="10.0.0.6",
                        help="Meta Quest 3 IP address")
    parser.add_argument("--use_unity", action="store_true",
                        help="Use Unity VR app instead of ADB")
    parser.add_argument("--vr_height", type=float, default=1.0,
                        help="VR headset height above ground (m)")
    parser.add_argument("--rotation_offset", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0],
                        metavar=("ROLL", "PITCH", "YAW"),
                        help="Hand-to-EE rotation offset in radians")
    parser.add_argument("--position_offset", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="Position offset to apply to hand poses")
    parser.add_argument("--position_signs", type=float, nargs=3,
                        default=[1.0, 1.0, 1.0],
                        metavar=("SX", "SY", "SZ"),
                        help="Axis sign flips for hand pose")
    parser.add_argument("--position_permutation", type=int, nargs=3,
                        default=[0, 1, 2],
                        metavar=("I0", "I1", "I2"),
                        help="Axis permutation for hand pose")

    args = parser.parse_args()
    collect_demos(args)


if __name__ == "__main__":
    main()
