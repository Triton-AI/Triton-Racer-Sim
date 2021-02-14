import time
from copy import copy
import cv2
import numpy as np
from PIL import Image
from numpy.core.defchararray import upper
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
        self.pt = cfg['perspective_transform']

        if self.upscale['enabled']:
            from cv2 import dnn_superres
            if self.upscale['scale'] == 2:
                model_path = 'ESPCN_x2.pb'
            elif self.upscale['scale'] == 4:
                model_path = 'ESPCN_x4.pb'
            else: raise Exception('Unsupported ai upscale factor.')

            self.sr = dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(model_path)
            self.sr.setModel("espcn", self.upscale['scale'])

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

        img = self.__crop(img)

        if self.color['enabled']:
            color_filtered_layers, img = self.__color_filter(img)
            #print(img.shape)
            layers.extend(color_filtered_layers)
            merge_instruction.extend(self.color['destination_channels'])
        if self.edge['enabled']:
            edge_filtered_layer = self.__edge_detection(img)
            layers.append(edge_filtered_layer)
            merge_instruction.append(self.edge['destination_channel'])
        if self.pt['enabled']:
            pts = np.array(self.pt['points'], dtype = "float32")
            img = self.__four_point_transform(img, pts)

        self.__merge(merge_instruction, img, layers)
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

        if self.color['display_method'] == 'overwrite':
            return output_layers, img
        elif self.color['display_method'] == 'overlay':
            for i in range(len(upper_bounds)):
                output_layers[i] = np.array(np.clip(cv2.accumulate(img[:,:,self.color['destination_channels'][i]], np.array(output_layers[i], dtype="float32")), 0, 255), dtype="uint8")
                for j in range(img.shape[2]):
                    if j != self.color['destination_channels'][i]:
                        img[:,:,j] = np.where(output_layers[i] == 255, 0, img[:,:,j])
            return output_layers, img

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
    

        original_shape = img.shape
        new_shape = original_shape[1] * self.upscale['scale'], original_shape[0] * self.upscale['scale']
        img = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, new_shape)

        #filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #img=cv2.filter2D(img,-1,filter)

        img = cv2.cvtColor(self.sr.upsample(img), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, tuple(self.upscale['target_resolution']))
        return img

    def __order_points(self, pts):
        # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect

    def __four_point_transform(self, image, pts):
        # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        # obtain a consistent order of the points and unpack them
        # individually
        # rect = self.__order_points(pts)
        rect = pts
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        return warped


    def onShutdown(self):
        self.running = False

    def getName(self):
        return 'Image Preprocessing'

class ImageResizer(Component):
    def __init__(self, cfg):
        super().__init__(inputs=['cam/img',], outputs=['cam/img',])
        self.target_res = cfg['img_h'], cfg['img_w']

    def step(self, *args):
        if args[0] is not None:
            return cv2.resize(args[0], self.target_res)

    def getName(self):
        return 'Image Resizer'