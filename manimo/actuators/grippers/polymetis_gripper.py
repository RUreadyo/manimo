import numpy as np
from gym import spaces
from manimo.actuators.grippers.gripper import Gripper
from omegaconf import DictConfig
from polymetis import GripperInterface


class PolymetisGripper(Gripper):
    """
    A class to control the Franka Emika Panda gripper
    """

    def __init__(self, gripper_cfg: DictConfig):
        """
        Initialize the gripper
        Args:
            gripper_cfg (DictConfig): The config for the gripper
        """
                
        self.config = gripper_cfg
        self._gripper_interface = GripperInterface(
            ip_address=gripper_cfg.server.ip_address,
            # port=gripper_cfg.server.port,
        )
        
        
        print(f"connection to gripper established!")
        self.action_space = spaces.Box(
            0.0,
            self._gripper_interface.metadata.max_width,
            (1,),
            dtype=np.float32,
        )

        # Flag to check if gripper is closed
        self.is_closed = False

    def _open_gripper(self):
        max_width = self._gripper_interface.metadata.max_width
        self._gripper_interface.goto(
            width=max_width,
            speed=self.config.speed,
            force=self.config.force,
        )

    def _close_gripper(self):
        self._gripper_interface.grasp(
            speed=self.config.speed, force=self.config.force
        )

    def step(self, action):
        obs = {}
        if action is not None:           
            # for pi zero
            # action is normalized from 0 to 1. Change to joint positions.
            action = np.clip(action, 0, 1)
            action = 1 - action
            
            if action < 0.5 :
                self._close_gripper()
            else:
                self._open_gripper()            


        obs["eef_gripper_action"] = action
        return obs

    def reset(self):
        """
        Reset the gripper to the initial state
        """
        self._open_gripper()
        print("gripper reset")
        return self.get_obs(), {}

    def get_obs(self):
        """
        Get the observations from the gripper
        Returns:
            ObsDict: The observations
        """
        # for pi-zero. 1 = closed, 0 = open.
        obs = {}
        width = self._gripper_interface.get_state().width
        
        obs["eef_gripper_width"] = width / self._gripper_interface.metadata.max_width
        obs["eef_gripper_width"] = np.clip(
            obs["eef_gripper_width"], 0, 1
        )
        obs["eef_gripper_width"] = 1- obs["eef_gripper_width"]

        
        return obs
