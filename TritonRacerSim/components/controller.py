import pygame
import os
from TritonRacerSim.components.component import Component
from enum import Enum

class DriveMode(Enum):
    HUMAN = 0
    AI_STEERING = 1
    AI = 2

class JoystickType(Enum):
    PS4 = 'ps4'
    PS3 = 'ps3'
    XBOX = 'xbox'

class Controller(Component):
    '''Generic base class for controllers'''
    def __init__(self):
        Component.__init__(self, threaded=True, outputs=['usr/steering', 'usr/throttle', 'usr/mode', 'usr/del_record', 'usr/toggle_record', 'usr/reset'])
        self.mode = DriveMode.HUMAN
        self.del_record = False
        self.toggle_record = False
        self.reset = False
        self.steering = 0.0
        self.throttle = 0.0
    def getName(self):
        return 'Generic Controller'

PS4_CONFIG={'steering_axis': 0, 'throttle_axis': 2, 'toggle_mode_but': 9, 'del_record_but': 3, 'toggle_record_but': 2, 'reset_but': 0}

class PygameJoystick(Controller):
    def __init__(self, joystick_type):
        Controller.__init__(self)
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.on = True
        print(f'Joystick name: {self.joystick.get_name()}')

        if (JoystickType(joystick_type) == JoystickType.PS4):
            self.joystick_map = PS4_CONFIG
        else:
             raise Exception('Unsupported joystick')

    def step(self, *args):
        to_return = (self.steering, self.throttle, str(self.mode), self.del_record, self.toggle_record, self.reset)
        self.del_record = False
        self.reset = False
        return to_return

    def thread_step(self):
        #function map
        switcher = {self.joystick_map['del_record_but']: self.__delRecord,
                    self.joystick_map['toggle_record_but']: self.__toggleRecord,
                    self.joystick_map['toggle_mode_but']: self.__toggleMode
                    }
        while self.on:
            self.steering = self.joystick.get_axis(self.joystick_map['steering_axis'])
            self.throttle = self.joystick.get_axis(self.joystick_map['throttle_axis'])

            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button in switcher:
                        switcher[event.button]()

    def onShutdown(self):
        self.on = False
        pygame.quit()

    def getName(self):
        return 'Pygame Joystick'

    def __toggleMode(self):
        if self.mode == DriveMode.HUMAN:
            self.mode = DriveMode.AI_STEERING
        elif self.mode == DriveMode.AI_STEERING:
            self.mode = DriveMode.AI
        elif self.mode == DriveMode.AI:
            self.mode = DriveMode.HUMAN
        return self.mode

    def __delRecord(self):
        self.del_record = True
        print ('Deleting records')

    def __toggleRecord(self):
        if self.toggle_record: 
            self.toggle_record = False
            print('Recording paused.')
        else:
            self.toggle_record = True
            print('Recording started.')

    def __reset(self):
        self.reset = True
        print('Car reset.')