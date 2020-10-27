import io
import socket
import time
import json

import numpy as np
from PIL import Image

from TritonRacerSim.components.component import Component
from gym_donkeycar.core.sim_client import SDClient
#import cv2

class ESP32CAM(Component, SDClient):
    def __init__(self, cfg):
        Component.__init__(self, inputs=['mux/steering', 'mux/throttle', 'mux/break'], outputs=['cam/img'], threaded=True)
        SDClient.__init__(self, cfg['ESP_ip'], cfg['ESP_port'], poll_socket_sleep_time=0.025)
        self.running = True

        self.left_pulse = cfg['calibrate_max_left_pwm']
        self.right_pulse = cfg['calibrate_max_right_pwm']
        self.neutral_steering_pulse = cfg['calibrate_neutral_steering_pwm']
        self.max_pulse = cfg['calibrate_max_forward_pwm']
        self.min_pulse = cfg['calibrate_max_reverse_pwm']
        self.zero_pulse = cfg['calibrate_zero_throttle_pwm']

    def onStart(self):
        '''
        self.serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serv.bind(('0.0.0.0', 9093))
        self.serv.listen(1)
        print("Seeking ESP32 CAM...")
        self.conn, addr = serv.accept()
        print(f"ESP32 CAM at {addr} is connected.")
        '''

    def step(self, *args):
        steering, throttle, breaking = args
        self.__command(self.validate(steering), self.validate(throttle), self.validate(breaking))

    def thread_step(self):
        while self.running:
            pass

    def onShutdown(self):
        """Shutdown"""
        self.running = False
        self.__command(self.neutral_steering_pulse, self.zero_pulse)

    def getName(self):
        return 'ESP32 CAM' 

    def on_msg_recv(self, j):
        msg_type = j['msg_type']
        if msg_type == 'image':
            pass
        elif msg_type == 'heartbeat':
            pass

    def validate(self, ctrl_num):
        if ctrl_num is None: return 0.0
        else: return ctrl_num

    def __command(self, steering=0.0, throttle = 0.0, breaking = 0.0, shutdown=False):
        """Send Instructions to ESP32"""
        msg = ''
        msg_dict = {}
        if not shutdown:
            msg_dict = {'msg_type': 'control', 'steering': steering, 'throttle': throttle, 'breaking': breaking}
        else:
            msg_dict = {'msg_type': 'shutdown',}
        msg = json.dumps(msg_dict) + '\n'
        self.send(msg)
'''
serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# address '0.0.0.0' or '' work to allow connections from other machines.  'localhost' disallows external connections.
# see https://www.raspberrypi.org/forums/viewtopic.php?t=62108
serv.bind(('0.0.0.0', 9093))
serv.listen(5)
print("Ready to accept 5 connections")


def create_image_from_bytes(image_bytes) -> Image.Image:
    stream = io.BytesIO(image_bytes)
    return Image.open(stream)


while True:
    conn, addr = serv.accept()
    array_from_client = bytearray()
    shape = None
    chunks_received = 0
    start = time.time()
    while True:
        # print('waiting for data')
        # Try 4096 if unsure what buffer size to use. Large transfer chunk sizes (which require large buffers) can cause corrupted results
        data = conn.recv(4096)
        if not data or data == b'tx_complete':
            break
        #elif shape is None:
            #shape_string += data.decode("utf-8")
            # Find the end of the line.  An index other than -1 will be returned if the end has been found because 
            # it has been received
            #if shape_string.find('\r\n') != -1:
                #width_index = shape_string.find('width:')
                #height_index = shape_string.find('height:')
                #width = int(shape_string[width_index + len('width:'): height_index])
                #height = int(shape_string[height_index + len('height:'): ])
                #shape = (width, height)
            #print("shape is {}".format(shape))
        else:
            chunks_received += 1
            array_from_client.extend(data)
            #time.sleep(0.01)

    print("chunks_received {}. Number of bytes {}".format(chunks_received, len(array_from_client)))
    img: Image.Image = create_image_from_bytes(array_from_client)
    image_array = np.asarray(img)
    cv2.imshow(image_array)
'''