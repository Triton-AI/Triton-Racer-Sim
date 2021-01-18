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

class PIDPilot(Component):
    def __init__(self, cfg):
        Component.__init__(self, inputs=['gym/speed', 'loc/cte'], outputs=['ai/steering', 'ai/throttle'], threaded=False)
        str_cfg = cfg['steering']
        spd_cfg = cfg['speed']

        # Input CTE to optimal racing line; Output steering
        self.str_pid = PID(str_cfg['kp'], str_cfg['ki'], str_cfg['kd'], setpoint=0.0, output_limits=(-1.0, 1.0), sample_time=0.01)

        # Input difference between the current speed and desired speed; OUtput throttle   
        self.spd_pid = PID(spd_cfg['kp'], spd_cfg['ki'], spd_cfg['kd'], setpoint=0.0, output_limits=(0.0, 1.0), sample_time=0.01)

        self.k = cfg['speed_limit_k']

    def step(self, *args):
        spd, cte = args
        if None not in args:
            str = self.str_pid(cte)
            
            d_spd = spd - self.k / abs(str) if str != 0.0 else -1
            thr = self.spd_pid(d_spd)
            cte_str = "{0:0.4f}".format(cte)
            print (f'CTE: {cte}, Str: {str}, Thr: {thr}\r', end='')
            return str, thr
        else:
            return 0.0, 0.0

    def getName(self):
        return 'PID Pilot'