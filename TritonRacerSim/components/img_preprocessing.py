import time
from copy import copy
import cv2
import numpy as np
from PIL import Image
from tensorflow.python.keras import constraints

from TritonRacerSim.components.component import Component

class ImgPreprocessing(Component):
    def __init__(self, cfg={}):
        super().__init__(inputs=['cam/img'], outputs=['cam/processed_img'], threaded=True)
        self.running = True
        self.to_process_img = None
        self.processed_img = None
        self.cfg = cfg
        self.contrast = cfg['contrast_enhancement']
        self.dy_bright = cfg['dynamic_brightness']
        self.color = cfg['color_filter']
        self.edge = cfg['edge_detection']
        self.upscale = cfg['ai_upscaling']

        if self.upscale['enabled']:
            from cv2 import dnn_superres
            if self.upscale['scale'] == 2:
                model_path = 'EDSR_x2.pb'
            elif self.upscale['scale'] == 4:
                model_path = 'EDSR_x4.pb'
            else: raise Exception('Unsupported ai upscale factor.')

            self.sr = dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(model_path)
            self.sr.setModel("edsr", self.upscale['scale'])

    def step(self, *args):
        img_arr = args[0]
        self.to_process_img = copy(img_arr) if img_arr is not None else None
        return self.processed_img,

    def thread_step(self):
        while (self.running):
            while self.to_process_img is None: # Waiting for new image
                time.sleep(0.005)

            # Copy the image, in case a new one is coming in
            img = self.to_process_img.copy()
            self.to_process_img = None
            img = self.__process(img)
            self.processed_img = img
            if self.cfg['preview_enabled']:
                cv2.imshow("Image Preprocessing Preview", cv2.cvtColor(self.processed_img,cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)           

    def __process(self, img):
        #print(img.shape)
        layers=[]
        merge_instruction=[]

        img = self.__trim_brightness_contrast(img)

        if self.upscale['enabled']:
            img = self.__upscale(img)
        if self.color['enabled']:
            color_filtered_layers = self.__color_filter(img)
            #print(img.shape)
            layers.extend(color_filtered_layers)
            merge_instruction.extend(self.color['destination_channels'])
        if self.edge['enabled']:
            edge_filtered_layer = self.__edge_detection(img)
            layers.append(edge_filtered_layer)
            merge_instruction.append(self.edge['destination_channel'])

        self.__merge(merge_instruction, img, layers)
        img = self.__crop(img)
        return img


    def __merge(self, instructions=[], destination=None, new_layers=None):
        # Replace the layers in destination according to instruction, preserving the untouched layers in the destination
        assert len(new_layers) == len(instructions)
        #print(destination.shape)
        for layer, instruction in zip(new_layers, instructions):
            destination[:,:,instruction] = layer
        #print(destination.shape)

    def __color_filter(self, img):
        hsv_img = cv2.cvtColor(img.copy(),cv2.COLOR_RGB2HSV)
        upper_bounds = self.color['hsv_upper_bounds']
        lower_bounds = self.color['hsv_lower_bounds']
        output_layers = []

        for i in range(len(upper_bounds)):
            mask = cv2.inRange(hsv_img, tuple(lower_bounds[i]), tuple(upper_bounds[i]))
            output_layers.append(mask)

        return output_layers

    def __edge_detection(self, img):
        threshold_a = self.edge['threshold_a']
        threshold_b = self.edge['threshold_b']
        return cv2.Canny(img, threshold_a, threshold_b)

    def __trim_brightness_contrast(self, img):
        dy_bright_enabled = self.dy_bright['enabled']
        contrast_enabled = self.contrast['enabled']

        if dy_bright_enabled or contrast_enabled:
            img_arr = img.astype(np.float32)
            if dy_bright_enabled:
                current_brightness = sum(list(cv2.mean(img[40:119,:,:])))
                brightness_baseline = self.dy_bright['baseline']
                delta = (brightness_baseline - current_brightness) / 3
                img_arr += delta
            if contrast_enabled:
                contrast = self.contrast['ratio']
                offset = self.contrast['offset']
                boost = self.contrast['channel_boost']
                for i in range(img.shape[2]):
                    img_arr[:,:,i] += boost[i]               
                img_arr = np.clip(img_arr, 0.0, 255.0)

                img_ave = np.average(img_arr, axis=2)
                mask = np.where(img_ave > offset, contrast, 1 / contrast)
                mask = np.swapaxes(np.tile(mask, (3, 1, 1)).T, 0, 1)
                img_arr = np.multiply(img_arr, mask)
            img_arr = np.clip(img_arr, 0, 255)
            img = img_arr.astype(np.uint8)

        return img

    def __crop(self, img):
        t, b, l, r = self.cfg['crop']
        img = img[t:, l:]
        if b > 0:
            img = img[:-b, :]
        if r > 0:
            img = img[:-r, :]
        return img

    def __upscale(self, img):
        return self.sr.upsample(img)

    def onShutdown(self):
        self.running = False

    def getName(self):
        return 'Image Preprocessing'