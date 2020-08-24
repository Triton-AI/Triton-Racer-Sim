import pygame
import time
import os
from TritonRacerSim.components.component import Component
from enum import Enum

class DriveMode(Enum):
    HUMAN = 'human'
    AI_STEERING = 'ai_steering'
    AI = 'ai'

class JoystickType(Enum):
    PS4 = 'ps4'
    PS3 = 'ps3'
    XBOX = 'xbox'
    G28 = 'g28'

class Controller(Component):
    '''Generic base class for controllers'''
    def __init__(self):
        Component.__init__(self, threaded=True, outputs=['usr/steering', 'usr/throttle', 'usr/breaking', 'usr/mode', 'usr/del_record', 'usr/toggle_record', 'usr/reset'])
        self.mode = DriveMode.HUMAN
        self.del_record = False
        self.toggle_record = False
        self.reset = False
        self.steering = 0.0
        self.throttle = 0.0
        self.breaking = 0.0
    def getName(self):
        return 'Generic Controller'

PS4_CONFIG={'steering_axis': 0, 'throttle_axis': 4, 'break_axis': 5, 'toggle_mode_but': 8, 'del_record_but': 2, 'toggle_record_but': 1, 'reset_but': 3, 'has_break': True}
G28_CONFIG={'steering_axis':0, 'throttle_axis': 2, 'break_axis' : 3,  'toggle_mode_but': 8, 'del_record_but': 2, 'toggle_record_but': 1, 'reset_but': 3, 'has_break': True}

class PygameJoystick(Controller):
    def __init__(self, joystick_type):
        Controller.__init__(self)
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.joystick.init()
        has_joystick = pygame.joystick.get_count()
        if has_joystick == 0:
            raise Exception('No joystick detected')
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.on = True
        print(f'Joystick name: {self.joystick.get_name()}')

        if JoystickType(joystick_type) == JoystickType.PS4:
            self.joystick_map = PS4_CONFIG
        elif JoystickType(joystick_type) == JoystickType.G28:
            self.joystick_map = G28_CONFIG
        else:
             raise Exception('Unsupported joystick')

    def step(self, *args):
        to_return = (self.steering, self.throttle, self.breaking, self.mode, self.del_record, self.toggle_record, self.reset)
        self.del_record = False
        self.reset = False
        return to_return

    def thread_step(self):
        #function map
        poll_interval = 10 # ms
        poll_interval /= 1000.0

        switcher = {self.joystick_map['del_record_but']: self.__delRecord,
                    self.joystick_map['toggle_record_but']: self.__toggleRecord,
                    self.joystick_map['toggle_mode_but']: self.__toggleMode,
                    self.joystick_map['reset_but']: self.__reset
                    }
        while self.on:
            self.steering = self.map_steering(self.joystick.get_axis(self.joystick_map['steering_axis']))
            self.throttle = self.map_throttle(self.joystick.get_axis(self.joystick_map['throttle_axis']))
            if self.joystick_map['has_break']:
                self.breaking = self.map_break(self.joystick.get_axis(self.joystick_map['break_axis']))

            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button in switcher:
                        switcher[event.button]()
            time.sleep(poll_interval)

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
        print (f'Mode: {self.mode}')
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

    def map_steering(self, val):
        return val

    def map_throttle(self, val):
        return val

    def map_break(self, val):
        return val

class G28DrivingWheel(PygameJoystick):
    def __init__(self):
        PygameJoystick.__init__(self, 'g28')

    def map_steering(self, val):
        val *= 5
        if val > 1:
            val = 1
        elif val < -1:
            val = -1
        return val
    
    def map_throttle(self, val):
        val = (val - 1) / 2 * -1
        return val

    def map_break(self, val):
        val = 1- ((val + 1) / 2)
        if val < 0.01:
            val = 0.0
        return val



class PS4(PygameJoystick):
    def __init__(self):
        PygameJoystick.__init__(self, 'ps4')

    def map_steering(self, val):
        return val

    def map_throttle(self, val):
        return val * -1

    def map_break(self, val):
        val = (val + 1) / 2
        if val < 0.2:
            val = 0.0
        return val

