
from TritonRacerSim.components.component import Component
import pygame, time
from pygame import camera
import numpy as np

class Camera(Component):
    def __init__(self, cfg):
        super().__init__(inputs=[], outputs=['cam/img', 'cam/original_img'], threaded=True)
        self.img_w = cfg['img_w']
        self.img_h = cfg['img_h']
        self.image_format = cfg['img_format']
        pygame.init()
        camera.init()
        cameras = camera.list_cameras()
        print ("Using camera %s ..." % cameras[cfg['idx']])
        self.webcam = camera.Camera(cameras[cfg['idx']], cfg['native_resolution'], cfg['img_format'])
        self.processed_frame = None
        self.original_frame = None
        self.on = True

    def onStart(self):
        """Called right before the main loop begins"""
        self.webcam.start()

    def step(self, *args):
        """The component's behavior in the main loop"""
        #if self.processed_frame is not None:
        #   print(self.processed_frame.shape)
        return self.processed_frame, self.original_frame

    def thread_step(self):
        """The component's behavior in its own thread"""

        while (self.on):
            #start_time = time.time()
            original_frame = self.webcam.get_image()
            processed_surface = pygame.transform.scale(original_frame, (self.img_w, self.img_h))
            #duration = time.time() - start_time
            #print (duration)
            self.original_frame = np.asarray(pygame.surfarray.array3d(original_frame))
            self.processed_frame = np.asarray(pygame.surfarray.array3d(processed_surface))
            # print (self.processed_frame.shape)
            #time.sleep(0.01)

    def onShutdown(self):
        """Shutdown"""
        self.on = False

    def getName(self):
        return 'USB Webcam (PyGame)'

class OpenCVCamera(Component):

    def __init__(self, cfg):
        import cv2
        super().__init__(inputs=[], outputs=['cam/img', 'cam/original_img'], threaded=True)
        self.img_w = cfg['img_w']
        self.img_h = cfg['img_h']
        self.native_w, self.native_h = cfg['native_resolution']
        self.image_format = cfg['img_format']

        self.cap = cv2.VideoCapture(cfg['idx'], cv2.CAP_V4L2)
        if self.image_format == 'YUV':
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
        elif self.image_format == 'MJPEG':
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.native_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.native_h)

        self.processed_frame = None
        self.original_frame = None
        self.on = True

    def onStart(self):
        """Called right before the main loop begins"""
        pass

    def step(self, *args):
        """The component's behavior in the main loop"""
        #if self.processed_frame is not None:
        #   print(self.processed_frame.shape)
        return self.processed_frame, self.original_frame

    def thread_step(self):
        """The component's behavior in its own thread"""
        import cv2
        while (self.on):
            rval, original_frame = self.cap.read()
            # if not rval: raise Exception("OpenCV Image Capture Error")
            if original_frame is not None:
                self.original_frame = original_frame
                original_frame = cv2.resize(original_frame, (self.img_w, self.img_h))
                self.processed_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            else:
                self.processed_frame = None
                self.original_frame = None

    def onShutdown(self):
        """Shutdown"""
        self.on = False
        self.cap.release()


    def getName(self):
        return 'USB Webcam (OpenCV)'