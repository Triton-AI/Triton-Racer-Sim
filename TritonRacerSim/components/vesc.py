import time
from pyvesc import VESCMessage, encode, decode
from pyvesc import VESC
from pyvesc.VESC.messages.setters import SetServoPosition, SetRPM
from pyvesc.VESC.messages.getters import GetValues

from TritonRacerSim.components.component import Component
from TritonRacerSim.utils.mapping import map_steering, map_throttle
from TritonRacerSim.components.controller import DriveMode

class VESC_(Component):
#class VESC_:
    def __init__(self, cfg):
        super().__init__(inputs=['mux/steering', 'mux/throttle', 'ai/speed', 'usr/mode'], outputs=['gym/speed'], threaded=True)
        self.running = True
        print("Connecting to VESC")
        self.v = VESC(serial_port=cfg['port'], baudrate=cfg['baudrate'], has_sensor=cfg['has_sensor'], start_heartbeat=cfg['enable_heartbeat'])
        print("VESC Connected")
        self.max_left = cfg['max_left_angle']
        self.max_right = cfg['max_right_angle']
        self.neutral_angle = cfg['neutral_angle']
        self.spd_ctl = cfg['ai_controls_speed']
        self.max_fwd = cfg['max_forward_rpm']
        self.max_rev = cfg['max_reverse_rpm']
        self.max_fwd_curr = cfg['max_forward_current']
        self.max_rev_curr = cfg['max_reverse_current']
        self.curr_rpm = 0.0

    def onStart(self):
        print("VESC Firmware Version: ", self.v.get_firmware_version())

    def step(self, *args):
        str, thr, ai_spd, mode = args
        if mode == DriveMode.AI and self.spd_ctl:
            if self.max_rev <= ai_spd <= self.max_fwd * 1.1:
                self.v.set_rpm(ai_spd)
            else:
                self.v.set_rpm(0)
        #elif self.curr_rpm > self.max_fwd:
        #    self.v.set_rpm(self.max_fwd)
        #elif self.curr_rpm < self.max_rev:
        #    self.v.set_rpm(self.max_rev)
        else:
            if -0.005 < thr < 0.005:
                self.send_current(0)
            else:
                self.send_current(int(map_throttle(thr, self.max_fwd_curr, 0, self.max_rev_curr)))
        self.v.set_servo(map_steering(str, self.max_left, self.neutral_angle, self.max_right))
        return self.curr_rpm,

    def send_servo_angle(self, angle):
        self.v.set_servo(angle)
    
    def send_rpm(self, rpm):
        self.v.set_rpm(rpm)

    def send_duty_cycle(self, dc):
        self.v.set_duty_cycle(dc)
    
    def send_current(self, curr):
        self.v.set_current(curr)

    def thread_step(self):
        while self.running:
            self.curr_rpm = self.v.get_rpm()
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
