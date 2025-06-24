import argparse
import numpy as np
import hydra
from manimo.scripts.manimo_loop import ManimoLoop
from manimo.utils.callbacks import BaseCallback
from manimo.teleoperation.teleop_agent import TeleopAgent
from manimo.utils.new_logger import DataLogger
from robobuf.buffers import ReplayBuffer 
import curses



class Teleop(BaseCallback):
    """
    Teleoperation callback.
    """

    def __init__(self, logger):
        super().__init__(logger)
        self.logger = logger
        self.keyboard = curses.initscr()
        curses.noecho()
        self.keyboard.nodelay(1)
        
        self.buttons = None

    

    def on_begin_traj(self, traj_idx):
        print(f"beginning new trajectory")
        pass

    def on_end_traj(self, traj_idx):
        if self.logger:
            self.logger.finish(traj_idx)
            pass

    def get_action(self, obs, pred_action=None):
        """
        Called at the end of each step.
        """
        self.buttons = self.keyboard.getch()        
        print(self.buttons)

        new_obs = obs.copy()
        
        if self.logger:
            self.logger.log(new_obs)


        return None

    def on_step(self, traj_idx, step_idx):
        
        if self.buttons:
            if self.buttons == ord('q'):
                curses.endwin()

                return True
            
        return False


def main():    
    replay_buffer = ReplayBuffer()
    logger = DataLogger(replay_buffer=replay_buffer, obs_keys=[], action_keys=[], storage_path=f"./human_demos/closelaptop")
    teleop_callback = Teleop(logger)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    manimo_loop = ManimoLoop(callbacks=[teleop_callback], T=1500, human=True)

    print(f'staring loop')
    
    manimo_loop.run()


if __name__ == "__main__":

    main()