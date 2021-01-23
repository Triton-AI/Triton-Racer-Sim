from TritonRacerSim.components.component import Component
from TritonRacerSim.utils.mapping import calcBreak, calcThrottleSim
from TritonRacerSim.components.controller import DriveMode
from simple_pid import PID
#from TritonRacerSim.utils.types import ModelType


# Speed Control Part calculates the output throttle for SIMULATED cars
# This part is not used for physical cars using the teensy microcontroller which performs its own speed calculations 
class SpeedControl(Component):
    def __init__(self, cfg):        # Inputs [current speed of car, speed predicted from the network based on training] Outputs [calculated throttle to send to car]
        super().__init__(inputs=['gym/speed', 'ai/speed', 'usr/mode', 'loc/break_indicator'], outputs=['ai/throttle', 'ai/breaking'], threaded=False)
        spd_cfg = cfg
        self.speed_control_threshold = spd_cfg['threshold']
        self.speed_control_reverse_multiplier = spd_cfg['reverse_multiplier']
        self.speed_control_break = spd_cfg['break']
        self.speed_control_break_multiplier = spd_cfg['break_multiplier']

    def step(self, *args):
        """
        Takes in current speed of car and the 'next' predicted speed from the model.
        Uses these values to calculate the necessary throttle to reach the predicted speed.
        Returns this calculated throttle value.
        """
        current_spd = args[0]
        predicted_spd = args[1]
        breaking = 0.0

        # Systems doesn't have image yet (probably just started)
        if args[0] is None:
            return 0.0, 0.0

        if args[3] == 1:
            predicted_spd *= 0.7

        # If system has image, we can calculate desired speed
        if  args[2] == DriveMode.AI_STEERING or args[2] == DriveMode.AI:
            # Calculate throttle using predicted speed from model
            throttle = calcThrottleSim(current_spd, predicted_spd * self.speed_control_threshold, self.speed_control_reverse_multiplier)

            # Check breaking
            if self.speed_control_break:
                throttle = 1.0 if predicted_spd - current_spd > 0.0 else 0.0
                breaking = calcBreak(current_spd, predicted_spd * self.speed_control_threshold, self.speed_control_break_multiplier)
            
            # print (f'Thr: {throttle}, Brk: {breaking} \r', end='')
            return throttle, breaking


        return 0.0, 0.0

    def getName(self):
        return 'Simple Speed Control'

class PIDSpeedControl(SpeedControl):
    def __init__(self, cfg):
        super().__init__(cfg)
        pid_cfg = cfg['pid']
        self.pid = PID(pid_cfg['kp'], pid_cfg['ki'], pid_cfg['kd'], setpoint=0.0, output_limits=(-1.0, 1.0), sample_time=0.05)

    def step(self, *args):
        current_spd = args[0]
        predicted_spd = args[1]
        breaking = 0.0 
        throttle = 0.0

        if None not in args:
            throttle = self.pid(current_spd - predicted_spd)
        return throttle, breaking
