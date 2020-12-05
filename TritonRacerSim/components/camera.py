
from TritonRacerSim.components.component import Component
import pygame, time
from pygame import camera
import numpy as np

class Camera(Component):
    
    def __init__(self, cfg):
        super().__init__(inputs=[], outputs=['cam/img', ], threaded=True)
        self.img_w = cfg['img_w']
        self.img_h = cfg['img_h']
        self.image_format = cfg['img_format']
        pygame.init()
        camera.init()
        cameras = camera.list_cameras()
        print ("Using camera %s ..." % cameras[cfg['idx']])
        self.webcam = camera.Camera(cameras[cfg['idx']], cfg['native_resolution'])
        self.processed_frame = None
        self.on = True

    def onStart(self):
        """Called right before the main loop begins"""
        self.webcam.start()

    def step(self, *args):
        """The component's behavior in the main loop"""
        return self.processed_frame,

    def thread_step(self):
        """The component's behavior in its own thread"""

        while (self.on):
            #start_time = time.time()
            original_frame = self.webcam.get_image()
            processed_surface = pygame.transform.scale(original_frame, (self.img_w, self.img_h))
            #duration = time.time() - start_time
            #print (duration)
            self.processed_frame = np.asarray(pygame.surfarray.array3d(processed_surface))
            #time.sleep(0.01)

    def onShutdown(self):
        """Shutdown"""
        self.on = False

    def getName(self):
        return 'Camera'