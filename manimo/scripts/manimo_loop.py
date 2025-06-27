import argparse
import hydra
import time
from manimo.environments.single_arm_env import SingleArmEnv


class ManimoLoop:
    def __init__(self, configs=None, callbacks=[], T=2000, human = False):
        self.callbacks = callbacks
        self.human = human
        if not configs:
            configs = ["sensors", "actuators", "env"]
            
            if human:
                # no actuator
                configs = ["sensors", None, "env"]
        
        self.T = T

        hydra.initialize(config_path="../conf", job_name="manimo_loop")

        env_configs = [
            hydra.compose(config_name=config_name) if config_name is not None else None for config_name in configs
        ]

        self.env = SingleArmEnv(*env_configs)


    def run(self):
        traj_idx = 0
        while True:
            import ipdb; ipdb.set_trace()
            print(f"Type c+enter! After it would start collecting trajectory {traj_idx} after 5 seconds...")
            time.sleep(5)
            
            obs, _ = self.env.reset()
            print(f"collecting trajectory {traj_idx} start!!!!")
            
            
            start_time = time.time()

            for callback in self.callbacks:
                callback.on_begin_traj(traj_idx)
            start_time = time.time()
            steps = 0
            for step_idx in range(self.T):

                steps += 1
                action = None
                for callback in self.callbacks:
                    new_action = callback.get_action(obs, pred_action=action)
                    if new_action is not None:
                        action = new_action

                if action is None:
                    if not self.human:
                        time.sleep(0.033)
                        continue
                
                if not self.human:
                    obs, _, _, _ = self.env.step(action)
                else : 
                    elapsed_time = time.time() - start_time
                    # make it 15Hz
                    if elapsed_time < 1/15:
                        time.sleep(1/15 - elapsed_time)
                    
                    obs = self.env.get_obs()
                print(f"env step took: {(time.time() - start_time)/steps}")
                finish = False
                for callback in self.callbacks:
                    finish = callback.on_step(traj_idx, step_idx)
                    # print(finish)
                    if finish:
                        break

                if finish:
                    break
                print(f"time per step: {(time.time() - start_time) / steps}")

            print(f"fps: {steps / (time.time() - start_time)}")

            for callback in self.callbacks:
                callback.on_end_traj(traj_idx)
            

            traj_idx += 1

def main():
    manimo_loop = ManimoLoop()
    

if __name__ == "__main__":
    main()
