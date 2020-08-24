from TritonRacerSim.components.component import Component
from tensorflow.python.keras.models import load_model
from os import path
import numpy as np
from PIL import Image
import tensorflow as tf
from TritonRacerSim.components.controller import DriveMode
from TritonRacerSim.utils.types import ModelType
import time
import os
import queue
import cv2

import matplotlib.pyplot as plt


class KerasPilot(Component):
    def __init__(self, model_path, model_type):
        inputs = ['cam/img', 'gym/speed', 'usr/mode']
        outputs = ['ai/steering', 'ai/throttle']
        self.model_type = model_type
        if model_type == ModelType.CNN_2D:
            pass

        Component.__init__(self, inputs=inputs, outputs=outputs, threaded=False)
        self.model = load_model(model_path, compile=True)
        self.model.summary()
        tf.keras.backend.set_learning_phase(0)
        self.on = True

        # from threading import Thread
        # t = Thread(target=self.prediction_thread, daemon=False)
        # t.start()

    
    def step(self, *args):
        if args[0] is None:
            return 0.0, 0.0
        if True or args[-1] == DriveMode.AI_STEERING or args[-1] == DriveMode.AI:
            img_arr = np.asarray(args[0],dtype=np.float32)
            img_arr /= 255
            #img = Image.fromarray(img_arr)
            #plt.imshow(img)
            #plt.show()
            #plt.clf()
            img_arr = img_arr.reshape((1,) + img_arr.shape)
            
            if self.model_type == ModelType.CNN_2D:
                # start_time = time.time()
                steering, throttle = self.model(img_arr)
                # print(f'Prediction time: {time.time() - start_time}')
                toreturn = steering.numpy()[0][0], throttle.numpy()[0][0]
                print(f'{toreturn}\r', end = '')
                return toreturn
            elif self.model_type == ModelType.CNN_2D_SPD_FTR:
                steering, throttle = self.model((img_arr, np.asarray((args[1] / 20,),dtype=np.float16)))
                toreturn = steering.numpy()[0][0] * 2 - 1, throttle.numpy()[0][0]
                print(f'{toreturn}\r', end = '')
                return toreturn
            elif self.model_type == ModelType.CNN_2D_SPD_CTL:
                steering, speed = self.model(img_arr)
                if (speed.numpy()[0][0] * 20 > args[1]):
                    throttle = 1.0
                else:
                    throttle = 0.0
                toreturn = steering.numpy()[0][0] * 2 - 1, throttle
                print(f'{toreturn}\r', end = '')
                return toreturn
        return 0.0, 0.0

    def onShutdown(self):
        self.on = False

    def getName(self):
        return 'Keras Pilot'


class PilotTester:
    def __init__(self):
        model_path = path.abspath('./try.h5')

        self.pilot = KerasPilot(model_path, ModelType.CNN_2D)
        pass

    def test(self):
        for i in range(6, 20):
            img_path = path.abspath('./data/valid/records_4/img_{}.jpg'.format(i))
            img_arr = np.asarray(Image.open(img_path))
            self.pilot.step(img_arr, DriveMode.AI)
        self.pilot.onShutdown()

#test = PilotTester()
#test.test()