import time
import os
import queue
from os import path
import numpy as np
from PIL import Image
import cv2
import torch.nn as nn
import torch
from torchvision import transforms

from TritonRacerSim.components.component import Component
from TritonRacerSim.components.controller import DriveMode
from TritonRacerSim.utils.types import ModelType
from TritonRacerSim.utils.mapping import calcBreak, calcThrottleSim, calcThrottlePhy
from TritonRacerSim.components.pytorch.pytorch_models import get_baseline_cnn


class PytorchPilot(Component):
    def __init__(self, cfg, model_path, model_type):
        inputs = ['cam/img', 'gym/speed', 'loc/segment', 'loc/break_indicator', 'usr/mode']
        outputs = ['ai/steering', 'ai/throttle', 'ai/breaking', 'ai/speed']
        self.model_type = model_type
        Component.__init__(self, inputs=inputs, outputs=outputs, threaded=False)
        if cfg['img_preprocessing']['enabled']:
            self.step_inputs[0] = 'cam/processed_img'

        model_cfg = cfg['ai_model']
        ''' NEW: TensorRT acceleration '''
        if model_cfg['use_tensorrt']:
            from torch2trt import TRTModule
            self.model = TRTModule()
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model = get_baseline_cnn()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.device = self.__get_device__()
            self.model = self.model.to(self.device)
            self.on = True
            self.trans = transforms.ToTensor()

        spd_cfg = cfg['speed_control']
        self.speed_control_threshold = spd_cfg['threshold']
        self.speed_control_reverse = spd_cfg['reverse']
        self.speed_control_break = spd_cfg['break']
        self.speed_control_reverse_multiplier = spd_cfg['reverse_multiplier']
        self.speed_control_break_multiplier = spd_cfg['break_multiplier']
        self.speed_mean = spd_cfg['train_speed_mean']
        self.speed_offset = spd_cfg['train_speed_offset']
        self.speed_categories = spd_cfg['categorical_speed_control']['intervals']

        smooth_cfg = cfg['ai_boost']['smooth_steering']
        self.smooth_steering = smooth_cfg['enabled']
        self.smooth_steering_threshold = smooth_cfg['threshold']
        self.thr_ctl_multiplier = cfg['ai_boost']['thr_ctl_multiplier']

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
            img_arr = np.expand_dims(img_arr, 0)
            
            if self.model_type == ModelType.CNN_2D_SPD_CTL:
                # print (img_arr.shape)
                img_arr = self.trans(img_arr).to(self.device)
                steering_and_speed = self.model(img_arr).detach().cpu().numpy()
                steering = self.__cap__(steering_and_speed[0][0])
                predicted_speed = (steering_and_speed[0][1] + self.speed_offset) * self.speed_mean
                if args[3] == 1: # Break indicator
                    predicted_speed *= 0.8
                # print (f'Spd: {real_spd}, Pred: {predicted_speed}, Str: {steering} \r', end='')
                #print (f'Thr: {throttle}, Brk: {breaking} \r', end='')
                steering = self.__smooth_steering(steering)
                return steering, None, None, predicted_speed

        return 0.0, 0.0, 0.0, 0.0

    def onStart(self):
        if self.cfg['img_preprocessing']['enabled']:
            print('[WARNING] Image preprocessing is enabled. Autopilot is fed with FILTERED image.')

    def onShutdown(self):
        self.on = False
            
    def getName(self):
        return 'Pytorch Pilot'

    def __cap__(self, val):
        if val < -1.0: val = -1.0
        elif val > 1.0: val = 1.0
        return val

    def __smooth_steering__(self, val):
        if self.smooth_steering:
            if val > self.smooth_steering_threshold:
                val = 1.0
            elif val < self.smooth_steering_threshold * -1:
                val = -1.0
        return val

    def __get_device__(self):
        if torch.cuda.is_available():
            print('Using CUDA.')
            return torch.device('cuda')
        print('CUDA not found. Using CPU for training.')       
        return torch.device('cpu')