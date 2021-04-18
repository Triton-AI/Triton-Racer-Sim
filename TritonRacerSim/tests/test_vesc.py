import time
from pyvesc import VESC

class VESC_:
    def __init__(self, port, baudrate=115200, inverted=False, has_sensor=False, start_heartbeat=True):
        print("Connecting to VESC")
        self.v = VESC(serial_port=port, baudrate=baudrate, has_sensor=has_sensor, start_heartbeat=start_heartbeat)
        print("VESC Connected")
        self.send_rpm(0)
        self.inverted = -1 if inverted else 1


    def print_firmware_version(self):
        print("VESC Firmware Version: ", self.v.get_firmware_version())

    def send_servo_angle(self, angle):
        self.v.set_servo(angle)
    
    def send_rpm(self, rpm):
        self.v.set_rpm(rpm)

    def send_duty_cycle(self, dc):
        # HACK. not used.
        self.v.set_duty_cycle(dc)
    
    def send_current(self, curr):
        self.v.set_current(curr)

    def get_rpm(self):
        return self.v.get_rpm() * self.inverted


if __name__ == "__main__":
    print('MAKE SURE YOUR CAR IS ON A STAND AND WHEELS CAN SPIN FREELY')
    input('Hit ENTER to continue...')
    v = VESC_(port="/dev/ttyACM0", baudrate=115200, inverted=False, has_sensor=False, start_heartbeat=True)

    forward_rpm = 10000
    backward_rpm = -10000
    steering_angle = 0.1 # in the range of [0, 1]

    v.send_servo_angle(steering_angle)

    v.send_rpm(forward_rpm)
    time.sleep(2)

    v.send_rpm(0)
    time.sleep(2)

    v.send_rpm(backward_rpm)
    time.sleep(2)

    v.send_rpm(0)
