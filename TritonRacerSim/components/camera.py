
from TritonRacerSim.components.component import Component
import pygame, time

class Camera(Component):
    
    def __init__(self, cfg):
        super().__init__(inputs=[], outputs=['cam/img', ], threaded=True)
        self.img_w = cfg['img_w']
        self.img_h = cfg['img_h']
        self.image_format = cfg['image_format']
        pygame.init()
        pygame.camera.init()
        cameras = pygame.camera.list_cameras()
        print ("Using camera %s ..." % cameras[cfg['cam_source']])
        self.webcam = pygame.camera.Camera(cameras[cfg['cam_source']], cfg['cam_resolution'])
        self.processed_frame = None
        self.on = True

    def onStart(self):
        """Called right before the main loop begins"""
        self.webcam.start()

    def step(self, *args):
        """The component's behavior in the main loop"""
        return self.processed_frame

    def thread_step(self):
        """The component's behavior in its own thread"""

        while (self.on):
            start_time = time.time()
            original_frame = self.webcam.get_image()
            self.processed_frame = pygame.transform.scale(original_frame, (self.img_w, self.img_h))
            duration = time.time() - start_time
            print (duration)
            import cv2, numpy
            cv2.imshow('frame', numpy.asarray(self.processed_frame))
            time.sleep(0.01)

    def onShutdown(self):
        """Shutdown"""
        self.on = False

    def getName(self):
        return 'Camera'