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
import numpy as np

from TritonRacerSim.components.component import Component
from TritonRacerSim.components.controller import DriveMode
class PIDPilot(Component):
    def __init__(self, cfg):
        Component.__init__(self, inputs=['gym/speed', 'loc/cte', 'usr/mode', 'loc/break_indicator', 'loc/heading', 'gym/telemetry', 'detection/objects_on_track'], outputs=['ai/steering', 'ai/throttle', 'ai/pid'], threaded=False)
        str_cfg = cfg['steering']
        spd_cfg = cfg['speed']

        # Input CTE to optimal racing line; Output steering
        self.str_pid = PID(str_cfg['kp'], str_cfg['ki'], str_cfg['kd'], setpoint=0.0, output_limits=(-1.0, 1.0), sample_time=0.05)

        # Input difference between the current speed and desired speed; Output throttle   
        self.spd_pid = PID(spd_cfg['kp'], spd_cfg['ki'], spd_cfg['kd'], setpoint=0.0, output_limits=(0.0, 1.0), sample_time=0.05)

        self.k = cfg['speed_limit_k']
        self.prev_mode = DriveMode.HUMAN

    def step(self, *args):
        spd, cte, mode, ind, heading_pred, tele, objects = args
        if None not in args[0:6]:
            if self.prev_mode != DriveMode.AI and mode == DriveMode.AI:
                self.str_pid.reset()
                self.spd_pid.reset()
            self.prev_mode = mode
            heading_diff = self.__calc_heading_difference(tele.yaw, heading_pred)
            heading_diff_normalized = heading_diff / (2 * np.pi) / 3
            cte = self.__correct_cte(cte, objects)
            cte_normalized = cte * 2 / 3

            str = self.str_pid(cte_normalized + heading_diff_normalized)
            
            d_spd = spd - (self.k / abs(str) + 5.0) if str != 0.0 else -1
            thr = self.spd_pid(d_spd)
            cte_str = "{0:0.4f}".format(cte)
            # print (f'CTE: {cte}, Str: {str}, Thr: {thr}\r', end='')
            if ind == 1:
                thr = 0.0
            return str, thr, self
        else:
            return 0.0, 0.0, self

    def __calc_heading_difference(self, curr_h, pred_h):
        # negative for turning left, positive for tunring right
        if pred_h > curr_h:
            tl = pred_h - curr_h
            tr = curr_h + 360 - pred_h
            return -tl if tl < tr else tr
        else:
            tl = 360 - curr_h + pred_h
            tr = curr_h - pred_h
            return -tl if tl < tr else tr

    def __correct_cte(self, cte, objects):
        clearance = 1
        return cte

        if objects is None or len(objects) == 0:
            return cte
        
        obj = objects[0]
        if abs(obj.cte) > clearance:
            return cte
        
        margin = abs(abs(obj.cte) - clearance)

        if (cte <= 0 and obj.cte >= 0) or (cte >= 0 and obj.cte <= 0):
            return cte - margin
        else:
            return cte + margin


    def getName(self):
        return 'PID Pilot'