import time
from pyvesc import VESC

from TritonRacerSim.components.component import Component
from TritonRacerSim.utils.mapping import map_steering, map_throttle
from TritonRacerSim.components.controller import DriveMode

class VESC_(Component):
#class VESC_:
    def __init__(self, cfg):
        # inputs are the topics it subscribes to. Output are the topics it publishes to.
        super().__init__(inputs=['mux/steering', 'mux/throttle', 'ai/speed', 'usr/mode', 'usr/breaking'], outputs=['gym/speed'], threaded=True)
        self.running = True
        print("Connecting to VESC")
        self.v = VESC(serial_port=cfg['port'], baudrate=cfg['baudrate'], has_sensor=cfg['has_sensor'], start_heartbeat=cfg['enable_heartbeat'])
        print("VESC Connected")
        self.send_rpm(0)
        self.max_left = cfg['max_left_angle']
        self.max_right = cfg['max_right_angle']
        self.neutral_angle = cfg['neutral_angle']
        self.spd_ctl = cfg['ai_controls_speed']
        self.max_fwd = cfg['max_forward_rpm']
        self.max_rev = cfg['max_reverse_rpm']
        self.max_fwd_curr = cfg['max_forward_current']
        self.max_rev_curr = cfg['max_reverse_current']
        self.curr_rpm = 0
        self.last_rpm = 0
        self.deadman_switch = cfg['break_as_deadman_switch']
        self.past_spds = []
        self.inverted = -1 if cfg['inverted'] else 1


    def onStart(self):
        print("VESC Firmware Version: ", self.v.get_firmware_version())


    def step(self, *args):
        '''
        step() is part of TritonRacer's Component API. It is called when the car iterates all the parts.
        args: a tuple of objects (messages from topics it subscribes to) as specified in the Component class initializer's inputs argument.
        return: a tuple of objects (messages to publish) as specified in the Component class initializer's output argument.
        '''
        str, thr, ai_spd, mode, breaking = args # steering in [-1, 1], throttle in [-1, 1], breaking in [0, 1]
        if mode == DriveMode.AI and self.spd_ctl:
            # In AI mode, we send steering and speed instruction. Also test if deadman switch (breaking trigger) is triggered.
            if self.deadman_switch and (breaking is not None) and breaking < 0.8:
                #print('Deadman Switch!')
                self.v.set_rpm(0)
                self.last_rpm = 0
                return self.curr_rpm, # return the real speed of the car

            # a naive limiter to discourage drastic speed changes
            if ai_spd > self.last_rpm + 300:
                ai_spd = self.last_rpm + 300
            elif ai_spd < self.last_rpm - 300:
                ai_spd = self.last_rpm - 300

            self.v.set_rpm(int(ai_spd))
            self.last_rpm = ai_spd

        else:
            # In human mode, we send steering and throttle instruction. A deadzone is applied.
            if (thr is not None  and -0.005 < thr < 0.005) or (mode == DriveMode.AI and breaking is not None and self.deadman_switch and breaking < 0.8):
                self.send_current(0) # let the car glide when zero throttle in human mode, or when deadman switch is triggered in AI mode.
            elif thr is not None:
                self.send_current(int(map_throttle(thr, self.max_fwd_curr, 0, self.max_rev_curr)))

        self.v.set_servo(map_steering(str, self.max_left, self.neutral_angle, self.max_right))
        return self.curr_rpm, # return the real speed of the car

    def send_servo_angle(self, angle):
        self.v.set_servo(angle)
    
    def send_rpm(self, rpm):
        self.v.set_rpm(rpm)

    def send_duty_cycle(self, dc):
        # HACK. not used.
        self.v.set_duty_cycle(dc)
    
    def send_current(self, curr):
        self.v.set_current(curr)

    def thread_step(self):
        '''
        thread_step() is part of TritonRacer's Component API. The thread is initialized by the car prior to driving.
        '''
        while self.running:
            self.curr_rpm = self.v.get_rpm() * self.inverted
            time.sleep(0.01) # sleep 10ms between each polling. prevent buffer overloading.

    def onShutdown(self):
        if self.v is not None:
            self.v.stop_heartbeat()
        self.running = False

    def getName(self):
        return 'VESC'


if __name__ == "__main__":
    v = VESC(serial_port="/dev/ttyACM0")
    print("Firmware: ", v.get_firmware_version())
