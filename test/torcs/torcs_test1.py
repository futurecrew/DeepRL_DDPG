import os
import time
import numpy as np
import test.torcs.snakeoil3_gym as snakeoil3

class TorcsTest():
    def __init__(self):
        self.vision = True
        self.throttle = True
        self.gear_change = True
        self.initial_run = True
        self.initial_reset = True
        
    def gogo(self):
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)
        
        self.reset()
        
        while True:
            self.client.get_servers_input()
            
            self.obs = self.client.S.d
             
            self.client.R.d['steer'] = -0.1
            self.client.R.d['accel'] = 0.5
            self.client.R.d['brake'] = 0.0
            self.client.R.d['gear'] = 1
            
            self.client.respond_to_server()
        

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        #self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        #return self.get_obs()

    def end(self):
        os.system('pkill torcs')

        
if __name__ == '__main__':
    TorcsTest().gogo()