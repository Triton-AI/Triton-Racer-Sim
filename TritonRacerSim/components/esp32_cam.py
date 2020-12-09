import base64
import io
import socket
import time
import json

import numpy as np
from PIL import Image

from TritonRacerSim.components.component import Component
from gym_donkeycar.core.sim_client import SDClient
import cv2

class ESP32CAM(Component, SDClient):
    def __init__(self, cfg):

        esp_cfg = cfg['esp32']
        calibrate_cfg = cfg['calibration']
        Component.__init__(self, inputs=['mux/steering', 'mux/throttle', 'mux/break'], outputs=['cam/img'], threaded=True)
        ip = esp_cfg['ip']
        port = esp_cfg['port']
        print(f"Connecting to {ip}:{port}...")
        SDClient.__init__(self, ip, port, poll_socket_sleep_time=0.025)
        self.running = True

        self.left_pulse = calibrate_cfg['max_left_pwm']
        self.right_pulse = calibrate_cfg['max_right_pwm']
        self.neutral_steering_pulse = calibrate_cfg['neutral_steering_pwm']
        self.max_pulse = calibrate_cfg['max_forward_pwm']
        self.min_pulse = calibrate_cfg['max_reverse_pwm']
        self.zero_pulse = calibrate_cfg['zero_throttle_pwm']
        self.img = None
        self.to_return = None

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
        return self.to_return,

    def thread_step(self):
        while self.running:
            if self.img is not None:
                img_arr = cv2.flip(cv2.flip(np.array(self.img),0),1)
                self.to_return = img_arr
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                cv2.imshow("ESP 32 Image", img_arr)
                cv2.waitKey(1) 
                time.sleep(0.05)

    def onShutdown(self):
        """Shutdown"""
        self.running = False
        self.__command(0.0, 0.0)

    def getName(self):
        return 'ESP32 CAM' 

    def on_msg_recv(self, j):
        msg_type = j['msg_type']
        if msg_type == 'image':
            self.img = self.create_image_from_bytes(j['data'])
        elif msg_type == 'heartbeat':
            pass

    def validate(self, ctrl_num):
        if ctrl_num is None: return 0.0
        else: return ctrl_num

    def __command(self, steering=0.0, throttle = 0.0, breaking = 0.0, shutdown=False):
        """Send Instructions to ESP32"""
        msg = ''
        msg_dict = {}
        from TritonRacerSim.utils.mapping import map_steering, map_throttle
        steering = int(map_steering(steering, self.left_pulse, self.neutral_steering_pulse, self.right_pulse))
        throttle = int(map_throttle(throttle, self.min_pulse, self.zero_pulse, self.max_pulse))
        if not shutdown:
            msg_dict = {'msg_type': 'control', 'steering': steering, 'throttle': throttle, 'breaking': breaking}
        else:
            msg_dict = {'msg_type': 'shutdown',}
        msg = json.dumps(msg_dict) + '\n'
        #print(msg)
        self.send(msg)

    def create_image_from_bytes(self, img_string) -> Image.Image:
        stream = io.BytesIO(base64.b64decode(img_string))
        return Image.open(stream)
'''

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