import time
import os
import queue
from os import path
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.python.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

from TritonRacerSim.components.component import Component
from TritonRacerSim.components.controller import DriveMode
from TritonRacerSim.utils.types import ModelType
from TritonRacerSim.utils.mapping import calcBreak, calcThrottleSim, calcThrottlePhy
from TritonRacerSim.components.keras_train import KerasResNetLSTM


class KerasPilot(Component):
    def __init__(self, cfg, model_path, model_type):
        inputs = ['cam/img', 'gym/speed', 'loc/segment', 'loc/break_indicator', 'usr/mode']
        outputs = ['ai/steering', 'ai/throttle', 'ai/breaking', 'ai/speed']
        self.model_type = model_type
        if model_type == ModelType.CNN_2D:
            pass
        Component.__init__(self, inputs=inputs, outputs=outputs, threaded=False)
        if cfg['img_preprocessing']['enabled']:
            self.step_inputs[0] = 'cam/processed_img'
        self.model = None
        if self.model_type == ModelType.LSTM:
            self.model = KerasResNetLSTM.get_model((cfg['cam']['img_h'], cfg['cam']['img_w'], 3), cfg['ai_model']['embedding_size'], 1)
            old_weights = load_model(model_path, compile=True).get_weights()
            self.model.set_weights(old_weights) # Change batch size to 1 to match with inference
        else:
            self.model = load_model(model_path, compile=True)
        self.model.summary()
        tf.keras.backend.set_learning_phase(0)
        self.on = True

        spd_cfg = cfg['speed_control']
        self.speed_control_threshold = spd_cfg['threshold']
        self.speed_control_reverse = spd_cfg['reverse']
        self.speed_control_break = spd_cfg['break']
        self.speed_control_reverse_multiplier = spd_cfg['reverse_multiplier']
        self.speed_control_break_multiplier = spd_cfg['break_multiplier']

        smooth_cfg = cfg['ai_boost']['smooth_steering']
        self.smooth_steering = smooth_cfg['enabled']
        self.smooth_steering_threshold = smooth_cfg['threshold']
        self.thr_ctl_multiplier = cfg['ai_boost']['thr_ctl_multiplier']

        model_cfg = cfg['ai_model']
        self.from_donkey = model_cfg['from_donkey']

        if self.smooth_steering:
            print('[WARNING] Smooth-Steering Enabled')

        self.cfg = cfg
        self.last_mode = None
        self.this_mode = None

    def step(self, *args):
        self.last_mode = self.this_mode
        self.this_mode = args[-1]
        if args[0] is None:
            return 0.0, 0.0, 0.0, 0.0
        if  args[-1] == DriveMode.AI_STEERING or args[-1] == DriveMode.AI:

            img_arr = np.asarray(args[0],dtype=np.float32)
            #img = Image.fromarray(img_arr)
            #plt.imshow(img)
            #plt.show()
            #plt.clf()
            img_arr = img_arr.reshape((1,) + img_arr.shape)
            
            if self.model_type == ModelType.CNN_2D:
                # start_time = time.time()
                img_arr /= 255
                steering_and_throttle = self.model(img_arr)
                # print(f'Prediction time: {time.time() - start_time}')
                if not self.from_donkey:
                    steering = self.__cap(steering_and_throttle.numpy()[0][0])
                    throttle = self.__cap(steering_and_throttle.numpy()[0][1])
                else:
                    steering = self.__cap(steering_and_throttle[0].numpy()[0][0])
                    throttle = self.__cap(steering_and_throttle[1].numpy()[0][0]) * self.thr_ctl_multiplier

                #steering = self.__smooth_steering(steering)
                # print (f'Str: {steering}, Thr: {throttle} \r', end='')

                return steering, throttle, 0.0, 0.0

            elif self.model_type == ModelType.CNN_2D_SPD_FTR:
                img_arr /= 255
                spd = np.asarray((args[1] / 20,), dtype=np.float32)
                spd = spd.reshape((1,) + spd.shape) 
                # print (img_arr.shape)
                steering_and_throttle = self.model((img_arr, spd))
                steering, throttle = self.__cap(steering_and_throttle.numpy()[0])

                steering = self.__smooth_steering(steering)

                return steering, throttle, 0.0, 0.0

            elif self.model_type == ModelType.CNN_2D_SPD_CTL: # Only used for simulated car, speed based control (speed_control.py part needed)
                # print (img_arr.shape)
                img_arr /= 255
                real_spd = args[1]
                steering_and_speed = self.model(img_arr)
                steering = self.__cap(steering_and_speed.numpy()[0][0])
                predicted_speed = (steering_and_speed.numpy()[0][1] + 1.0) * 10.0

                # print (f'Spd: {real_spd}, Pred: {predicted_speed}, Str: {steering} \r', end='')
                #print (f'Thr: {throttle}, Brk: {breaking} \r', end='')
                steering = self.__smooth_steering(steering)
                return steering, None, None, predicted_speed

            elif self.model_type == ModelType.CNN_2D_FULL_HOUSE: # Only used for simulated car, speed based control (speed_control.py part needed)
                # print (args[1], args[2], args[3])
                img_arr /= 255
                real_spd = args[1]
                features = np.asarray((args[3],))
                features = features.reshape((1,) + features.shape)
                steering_and_speed = self.model((img_arr, features))
                steering = self.__cap(steering_and_speed.numpy()[0][0])
                predicted_speed = (steering_and_speed.numpy()[0][1] + 1.0) * 10.0
                # print (f'Spd: {real_spd}, Pred: {predicted_speed} \r', end='')

                steering = self.__smooth_steering(steering)
                #print (f'Str: {steering}, Thr: {throttle}, Brk: {breaking} \r', end='')
                
                return steering, None, None, predicted_speed
                
            elif self.model_type == ModelType.LSTM:
                if self.last_mode != self.this_mode:
                    print("Resetting States")
                    self.model.get_layer('decoder').reset_states()
                img_arr = preprocess_input(img_arr)
                spd = np.asarray((args[1],), dtype=np.float32) 
                # print (img_arr.shape)
                steering_and_throttle = self.model((img_arr, spd))
                steering, throttle = self.__cap(steering_and_throttle.numpy()[0][0]), self.__cap(steering_and_throttle.numpy()[0][1])

                steering = self.__smooth_steering(steering)
                # print (f'Str: {steering}, Thr: {throttle}\r', end='')
                return steering, throttle, 0.0, 0.0

            elif self.model_type == ModelType.CNN_2D_SPD_CTL_BREAK_INDICATION:
                img_arr /= 255
                ind = np.asarray((args[1] / 20,), dtype=np.float32)
                ind = ind.reshape((1,) + ind.shape) 
                # print (img_arr.shape)
                steering_and_throttle = self.model((img_arr, ind))
                steering = self.__cap(steering_and_throttle.numpy()[0][0])
                speed = steering_and_throttle.numpy()[0][1]
                speed = (speed + 1.0) * 20

                steering = self.__smooth_steering(steering)

                return steering, None, None, speed

        return 0.0, 0.0, 0.0, 0.0

    def onStart(self):
        if self.cfg['img_preprocessing']['enabled']:
            print('[WARNING] Image preprocessing is enabled. Autopilot is fed with FILTERED image.')

    def onShutdown(self):
        self.on = False
            
    def getName(self):
        return 'Keras Pilot'

    def __cap(self, val):
        if val < -1.0: val = -1.0
        elif val > 1.0: val = 1.0
        return val

    def __smooth_steering(self, val):
        if self.smooth_steering:
            if val > self.smooth_steering_threshold:
                val = 1.0
            elif val < self.smooth_steering_threshold * -1:
                val = -1.0
        return val


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