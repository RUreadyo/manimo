import time

import grpc
import numpy as np
import torch
import torchcontrol as toco
from manimo.actuators.arms.arm import Arm
from manimo.actuators.arms.robot_ik.robot_ik_solver import RobotIKSolver

# from manimo.actuators.arms.moma_arm import MujocoArmModel
from manimo.actuators.controllers.policies import CartesianPDPolicy, JointPDPolicy
from manimo.teleoperation.teleop_agent import quat_add
from manimo.utils.helpers import Rate
from manimo.utils.types import ActionSpace, IKMode
from manimo.utils.transformations import *

from omegaconf import DictConfig
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R


class CustomRobotInterace(RobotInterface):
    def get_ee_pose(self):
        eef_position, eef_orientation = super().get_ee_pose()
        r = toco.transform.Rotation.from_quat(eef_orientation)
        pos_offset = r.apply(
            torch.Tensor([0, 0, 0.0])
        )  # we do not use offset for DemoDiffusion. Instead, we apply offset in kinematic retargeting.
        return eef_position + pos_offset, eef_orientation


def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler


class FrankaArm(Arm):
    def __init__(self, arm_cfg: DictConfig):
        self.config = arm_cfg
        self.action_space = ActionSpace(arm_cfg.action_space)
        self.delta = arm_cfg.delta
        self.hz = 15 
        self.ik_mode = IKMode(arm_cfg.ik_mode)
        self.JOINT_LIMIT_MIN = arm_cfg.joint_limit_min
        self.JOINT_LIMIT_MAX = arm_cfg.joint_limit_max
        
        
        self.robot = CustomRobotInterace(
            ip_address=self.config.robot_ip, enforce_version=False
        )
        self.robot.hz = arm_cfg.hz
        self.kq = arm_cfg.kq
        self.kqd = arm_cfg.kqd
        self.home = (
            arm_cfg.home
            if arm_cfg.home is not None
            else self.robot.get_joint_positions()
        )
        self._ik_solver = RobotIKSolver()
        self.reset()
        

    def set_home(self, home):
        self.home = home

    def connect(self, policy=None, wait=2):
        if policy is None:
            policy = self._default_policy(self.action_space)
        self.policy = policy
        self.robot.send_torch_policy(policy, blocking=False)
        time.sleep(wait)

    def get_robot_state(self):
        robot_state = self.robot.get_robot_state()
        gripper_position = 0
        
        pos, quat = self.robot.robot_model.forward_kinematics(
            torch.Tensor(robot_state.joint_positions)
        )

        cartesian_position = (
            pos.tolist() + quat_to_euler(quat.numpy()).tolist()
        )

        state_dict = {
            "cartesian_position": cartesian_position,
            "gripper_position": gripper_position,
            "joint_positions": list(robot_state.joint_positions),
            "joint_velocities": list(robot_state.joint_velocities),
            "joint_torques_computed": list(robot_state.joint_torques_computed),
            "prev_joint_torques_computed": list(
                robot_state.prev_joint_torques_computed
            ),
            "prev_joint_torques_computed_safened": list(
                robot_state.prev_joint_torques_computed_safened
            ),
            "motor_torques_measured": list(robot_state.motor_torques_measured),
            "prev_controller_latency_ms": (
                robot_state.prev_controller_latency_ms
            ),
            "prev_command_successful": robot_state.prev_command_successful,
        }

        timestamp_dict = {
            "robot_timestamp_seconds": robot_state.timestamp.seconds,
            "robot_timestamp_nanos": robot_state.timestamp.nanos,
        }

        return state_dict, timestamp_dict

    def reset(self):
        self._go_home()
        self.connect()

        obs = self.get_obs()
        return obs, {}

    def _go_home(self):
        home = torch.Tensor(self.home)

        # Create policy instance
        q_initial = self.robot.get_joint_positions()
        waypoints = toco.planning.generate_joint_space_min_jerk(
            start=q_initial, goal=home, time_to_go=4, hz=self.hz
        )
        rate = Rate(self.hz)
        joint_positions = [waypoint["position"] for waypoint in waypoints]

        q_initial = self.robot.get_joint_positions()
        kq = torch.Tensor(self.robot.metadata.default_Kq)
        kqd = torch.Tensor(self.robot.metadata.default_Kqd)
        policy = JointPDPolicy(
            desired_joint_pos=q_initial,
            kq=kq,
            kqd=kqd,
        )
        self.robot.send_torch_policy(policy, blocking=False)
        rate.sleep()
        for joint_position in joint_positions:
            self.robot.update_current_policy({"q_desired": joint_position})
            rate.sleep()

    def _default_policy(self, action_space, kq_ratio=1.0, kqd_ratio=1.0):
        q_initial = self.robot.get_joint_positions()
        kq = kq_ratio * torch.Tensor(self.kq)
        kqd = kqd_ratio * torch.Tensor(self.kqd)
        kx = torch.Tensor(self.robot.metadata.default_Kx)
        kxd = torch.Tensor(self.robot.metadata.default_Kxd)

        if action_space == ActionSpace.Joint:
            return toco.policies.HybridJointImpedanceControl(
                joint_pos_current=q_initial,
                Kq=kq,
                Kqd=kqd,
                Kx=kx,
                Kxd=kxd,
                robot_model=self.robot.robot_model,
                ignore_gravity=True,
            )
        elif action_space == ActionSpace.Cartesian:
            if self.ik_mode == IKMode.Polymetis:
                return CartesianPDPolicy(q_initial, True, kq, kqd, kx, kxd)
            elif self.ik_mode == IKMode.DMControl:
                return toco.policies.HybridJointImpedanceControl(
                    joint_pos_current=q_initial,
                    Kq=kq,
                    Kqd=kqd,
                    Kx=kx,
                    Kxd=kxd,
                    robot_model=self.robot.robot_model,
                    ignore_gravity=True,
                )
        elif action_space == ActionSpace.JointOnly:
            return toco.policies.JointImpedanceControl(
                joint_pos_current=q_initial,
                Kp=kq,
                Kd=kqd,
                robot_model=self.robot.robot_model,
                ignore_gravity=True,
            )

    def _get_desired_pos_quat(self, eef_pose):

        if self.delta:
            ee_pos_cur, ee_quat_cur = self.robot.get_ee_pose()
            ee_pos_desired = ee_pos_cur + torch.Tensor(eef_pose[:3])

            # add two quaternions
            ee_quat_desired = torch.Tensor(quat_add(ee_quat_cur, eef_pose[3:]))
        else:
            ee_pos_desired = torch.Tensor(eef_pose[:3])
            ee_quat_desired = torch.Tensor(eef_pose[3:])


        return ee_pos_desired, ee_quat_desired

    def _apply_joint_commands(self, q_desired):
        q_des_tensor = np.array(q_desired)
        q_des_tensor = torch.tensor(
            np.clip(q_des_tensor, self.JOINT_LIMIT_MIN, self.JOINT_LIMIT_MAX)
        )
        try:
            self.robot.update_current_policy(
                {"joint_pos_desired": q_des_tensor.float()}
            )
        except grpc.RpcError:
            import ipdb; ipdb.set_trace()
            self.reset()

    def _apply_eef_commands(self, eef_pose, wait_time=3):
        ee_pos_desired, ee_quat_desired = self._get_desired_pos_quat(eef_pose)

        joint_pos_cur = self.robot.get_joint_positions()
        joint_pos_desired, success = self.robot.solve_inverse_kinematics(
            ee_pos_desired, ee_quat_desired, joint_pos_cur
        )
        joint_pos_desired = joint_pos_desired.numpy()

        update_success = True
        try:
            joint_pos_desired = torch.tensor(
                np.clip(
                    joint_pos_desired,
                    self.JOINT_LIMIT_MIN,
                    self.JOINT_LIMIT_MAX,
                )
            )
            self.robot.update_current_policy(
                {"joint_pos_desired": joint_pos_desired.float()}
            )
        except grpc.RpcError:
            self.robot.send_torch_policy(self.policy, blocking=False)
            update_success = False
            time.sleep(wait_time)

        return update_success
    
    def _apply_eef_commands_soft(self, eef_pose, time_to_go = 3):
        ee_pos_desired, ee_quat_desired = self._get_desired_pos_quat(eef_pose)
        
        joint_pos_cur = self.robot.get_joint_positions()
        joint_pos_desired, success = self.robot.solve_inverse_kinematics(
            ee_pos_desired, ee_quat_desired, joint_pos_cur
        )
        joint_pos_desired = joint_pos_desired.numpy()
        
        self.soft_ctrl(action_target= joint_pos_desired, time_to_go=time_to_go)
        
        
        

    def step(self, action):
        action_obs = {"delta": self.delta, "action": action.copy()}


        if self.action_space == ActionSpace.Cartesian:
            if self.ik_mode == IKMode.Polymetis:
                self._apply_eef_commands(action)

            elif self.ik_mode == IKMode.DMControl:
                ee_pos_current, ee_quat_current = self.robot.get_ee_pose()
                cur_joint_positions = self.robot.get_joint_positions().numpy()
                unscaled_action = action / 15
                ee_pos_desired, ee_quat_desired = self._get_desired_pos_quat(
                    unscaled_action
                )

                robot_state = self.get_robot_state()[0]
                joint_velocity = (
                    self._ik_solver.cartesian_velocity_to_joint_velocity(
                        unscaled_action, robot_state=robot_state
                    )
                )

                joint_delta = self._ik_solver.joint_velocity_to_delta(
                    joint_velocity
                )
                desired_joint_action = (
                    joint_delta + self.robot.get_joint_positions().numpy()
                )

                command_status = self._apply_joint_commands(
                    desired_joint_action
                )

            action_obs["joint_action"] = desired_joint_action
            action_obs["ee_pos_action"] = ee_pos_desired.numpy()
            action_obs["ee_quat_action"] = ee_quat_desired.numpy()
            action_obs["joint_velocity"] = joint_velocity

        elif self.action_space == ActionSpace.Joint:
            if self.delta:
                joint_delta = self._ik_solver.joint_velocity_to_delta(action)
                action = joint_delta + self.robot.get_joint_positions().numpy()
            
            self.robot.update_desired_joint_positions(torch.Tensor(action))
            # command_status = self._apply_joint_commands(action)
            
            
            action_obs["joint_action"] = action
            (
                ee_pos_desired,
                ee_quat_desired,
            ) = self.robot.robot_model.forward_kinematics(torch.tensor(action))
            action_obs["ee_pos_action"] = ee_pos_desired.numpy()
            action_obs["ee_quat_action"] = ee_quat_desired.numpy()

        elif self.action_space == ActionSpace.JointOnly:
            command_status = self._apply_joint_commands(action)
            action_obs["joint_action"] = action
            (
                ee_pos_desired,
                ee_quat_desired,
            ) = self.robot.robot_model.forward_kinematics(action)
            action_obs["ee_pos_action"] = ee_pos_desired.numpy()
            action_obs["ee_quat_action"] = ee_quat_desired.numpy()
        return action_obs

    def get_obs(self):
        obs = {}
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        eef_position, eef_orientation = self.robot.get_ee_pose()
        obs["q_pos"] = joint_positions.numpy()
        obs["q_vel"] = joint_velocities.numpy()
        obs["eef_pos"] = eef_position.numpy()
        obs["eef_rot"] = eef_orientation.numpy()
        return obs


    def soft_ctrl(self, action_target, time_to_go=4):
        goal = torch.Tensor(action_target)
        
        # Create policy instance
        q_initial = self.robot.get_joint_positions()
        waypoints = toco.planning.generate_joint_space_min_jerk(
            start=q_initial, goal=goal, time_to_go=time_to_go, hz=self.hz
        )
        rate = Rate(self.hz)
        joint_positions = [waypoint["position"] for waypoint in waypoints]

        q_initial = self.robot.get_joint_positions()
        kq = torch.Tensor(self.robot.metadata.default_Kq)
        kqd = torch.Tensor(self.robot.metadata.default_Kqd)
        policy = JointPDPolicy(
            desired_joint_pos=q_initial,
            kq=kq,
            kqd=kqd,
        )
        self.robot.send_torch_policy(policy, blocking=False)
        rate.sleep()
        for joint_position in joint_positions:
            self.robot.update_current_policy({"q_desired": joint_position})
            rate.sleep()
            
        self.connect()
        



    def create_action_dict(self, action, action_space, gripper_action_space=None, robot_state=None):
        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        if robot_state is None:
            robot_state = self.get_robot_state()[0]
        action_dict = {"robot_state": robot_state}
        velocity = "velocity" in action_space

        if gripper_action_space is None:
            gripper_action_space = "velocity" if velocity else "position"
        assert gripper_action_space in ["velocity", "position"]
            

        if gripper_action_space == "velocity":
            action_dict["gripper_velocity"] = action[-1]
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(action[-1])
            gripper_position = robot_state["gripper_position"] + gripper_delta
            action_dict["gripper_position"] = float(np.clip(gripper_position, 0, 1))
        else:
            action_dict["gripper_position"] = float(np.clip(action[-1], 0, 1))
            gripper_delta = action_dict["gripper_position"] - robot_state["gripper_position"]
            gripper_velocity = self._ik_solver.gripper_delta_to_velocity(gripper_delta)
            action_dict["gripper_delta"] = gripper_velocity

        if "cartesian" in action_space:
            if velocity:
                action_dict["cartesian_velocity"] = action[:-1]
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(action[:-1])
                action_dict["cartesian_position"] = add_poses(
                    cartesian_delta, robot_state["cartesian_position"]
                ).tolist()
            else:
                action_dict["cartesian_position"] = action[:-1]
                cartesian_delta = pose_diff(action[:-1], robot_state["cartesian_position"])
                cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)
                action_dict["cartesian_velocity"] = cartesian_velocity.tolist()

            action_dict["joint_velocity"] = self._ik_solver.cartesian_velocity_to_joint_velocity(
                action_dict["cartesian_velocity"], robot_state=robot_state
            ).tolist()
            joint_delta = self._ik_solver.joint_velocity_to_delta(action_dict["joint_velocity"])
            action_dict["joint_position"] = (joint_delta + np.array(robot_state["joint_positions"])).tolist()

        if "joint" in action_space:
            # NOTE: Joint to Cartesian has undefined dynamics due to IK
            if velocity:
                action_dict["joint_velocity"] = action[:-1]
                joint_delta = self._ik_solver.joint_velocity_to_delta(action[:-1])
                action_dict["joint_position"] = (joint_delta + np.array(robot_state["joint_positions"])).tolist()
            else:
                action_dict["joint_position"] = action[:-1]
                joint_delta = np.array(action[:-1]) - np.array(robot_state["joint_positions"])
                joint_velocity = self._ik_solver.joint_delta_to_velocity(joint_delta)
                action_dict["joint_velocity"] = joint_velocity.tolist()
        
        
        return action_dict
    
    


