""" 
An alternative approach to neural network AI pilot
using pid control to follow a predefined race line.

Usage:
1. Drive around the track, carefully following the optimal race line;
2. Process into a track data;
3. Turn on location tracker in myconfig;
4. Set model_type to "pid" in myconfig;
5. Drive.
"""
from simple_pid import PID

from TritonRacerSim.components.component import Component
from TritonRacerSim.components.controller import DriveMode
class PIDPilot(Component):
    def __init__(self, cfg):
        Component.__init__(self, inputs=['gym/speed', 'loc/cte', 'usr/mode', 'loc/break_indicator'], outputs=['ai/steering', 'ai/throttle', 'ai/pid'], threaded=False)
        str_cfg = cfg['steering']
        spd_cfg = cfg['speed']

        # Input CTE to optimal racing line; Output steering
        self.str_pid = PID(str_cfg['kp'], str_cfg['ki'], str_cfg['kd'], setpoint=0.0, output_limits=(-1.0, 1.0), sample_time=0.05)

        # Input difference between the current speed and desired speed; Output throttle   
        self.spd_pid = PID(spd_cfg['kp'], spd_cfg['ki'], spd_cfg['kd'], setpoint=0.0, output_limits=(0.0, 1.0), sample_time=0.05)

        self.k = cfg['speed_limit_k']
        self.prev_mode = DriveMode.HUMAN

    def step(self, *args):
        spd, cte, mode, ind = args
        if None not in args:
            if self.prev_mode != DriveMode.AI and mode == DriveMode.AI:
                self.str_pid.reset()
                self.spd_pid.reset()
            self.prev_mode = mode

            str = self.str_pid(cte)
            
            d_spd = spd - self.k / abs(str) if str != 0.0 else -1
            thr = self.spd_pid(d_spd)
            cte_str = "{0:0.4f}".format(cte)
            # print (f'CTE: {cte}, Str: {str}, Thr: {thr}\r', end='')
            if ind == 1:
                thr = 0.0
            return str, thr, self
        else:
            return 0.0, 0.0, self

    def getName(self):
        return 'PID Pilot'