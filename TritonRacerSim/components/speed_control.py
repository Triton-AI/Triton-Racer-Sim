from TritonRacerSim.components.component import Component
from TritonRacerSim.utils.mapping import calcBreak, calcThrottleSim
from TritonRacerSim.components.controller import DriveMode
#from TritonRacerSim.utils.types import ModelType


# Speed Control Part calculates the output throttle for SIMULATED cars
# This part is not used for physical cars using the teensy microcontroller which performs its own speed calculations 
class SpeedControl(Component):
    def __init__(self, cfg):        # Inputs [current speed of car, speed predicted from the network based on training] Outputs [calculated throttle to send to car]
        super().__init__(inputs=['gym/speed', 'ai/speed', 'usr/mode'], outputs=['ai/throttle', 'ai/breaking'], threaded=False)
        spd_cfg = cfg['speed_control']
        self.speed_control_threshold = spd_cfg['threshold']
        self.speed_control_reverse_multiplier = spd_cfg['reverse_multiplier']
        self.speed_control_break = spd_cfg['break']
        self.on = True

    def onStart(self):
        """What should I do right at the launch of the car?"""
        pass

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

        # If system has image, we can calculate desired speed
        if  args[-1] == DriveMode.AI_STEERING or args[-1] == DriveMode.AI:
            # Calculate throttle using predicted speed from model
            throttle = calcThrottleSim(current_spd, predicted_spd * self.speed_control_threshold, self.speed_control_reverse_multiplier)

            # Check breaking
            if self.speed_control_break:
                throttle = 1.0 if predicted_speed - real_spd > 0.0 else 0.0
                breaking = calcBreak(real_spd, predicted_speed * self.speed_control_threshold, self.speed_control_break_multiplier)
            
            print (f'Thr: {throttle}, Brk: {breaking} \r', end='')
            return throttle, breaking


        return 0.0, 0.0

    def thread_step(self):
        """The component's behavior in its own thread"""
        pass

    def onShutdown(self):
        """What to do at shutdown?"""
        self.on = False

    def getName(self):
        return 'Generic Custom Component'
