
from TritonRacerSim.components.component import Component
import numpy as np
from rplidar import RPLidar
import serial

class Rplidar(Component):
    
    def __init__(self, cfg):
        super().__init__(inputs=[], outputs=['lidar', ], threaded=True)
        self.port = cfg['lidar']
        self.distance = [] # List distance, will be placed in datastorage
        self.angles = [] # List angles, will be placed in datastorage
        self.lidar = RPLidar(self.port)
        self.on = True
        self.lidar.clear_input()

    def onStart(self):
        """Called right before the main loop begins"""
        self.lidar.connect()
        self.lidar.start_motor()
        

    def step(self, *args):
        return self.distance, self.angles,


    def thread_step(self):
        """The component's behavior in its own thread"""
        scans = self.lidar.iter_scans()
        while self.on:
            try:
                for distance, angle in scans:
                    for item in distance: # Just pick either distance or angle since they have the same amount in list
                        self.distance = [item[2]] # Want to get the 3rd item which gives the distance from scans
                        self.angles = [item[1]] # Want to get the 2nd item which gives the angle from scans
            except serial.SerialException:
                print('Common exception when trying to thread and shutdown')

    def onShutdown(self):
        """Shutdown"""
        self.on = False
        self.lidar.stop()
        self.lidar.stop_motor()
        self.lidar.disconnect()

    def getName(self):
        return 'Rplidar'


class LidarPlot(Component):
    def __init__(self, inputs = ['img', 'distance', theta]= , outputs = [], threaded = True):
        """The name of input and output values must be provided as strings (e.g. 'speed', 'throttle')"""
        self.step_inputs = inputs.copy()
        self.step_outputs = outputs.copy()
        self.threaded = threaded

    def onStart(self):
        """Called right before the main loop begins"""
        pass

    def step(self, *args):
        """The component's behavior in the main loop"""
        pass

    def thread_step(self):
        """The component's behavior in its own thread"""
        pass

    def onShutdown(self):
        """Shutdown"""
        pass

    def getName(self):
        return 'LidarPlot'