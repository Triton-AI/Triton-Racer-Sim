from TritonRacerSim.components.component import Component
from TritonRacerSim.components.controller import DriveMode
from threading import Thread
import time

class ControlMultiplexer(Component):
    '''Switch user or ai control based on mode. also controls ai launch'''
    def __init__(self, cfg={}):
        Component.__init__(self, inputs=['usr/mode', 'usr/steering', 'usr/throttle', 'usr/breaking', 'ai/steering', 'ai/throttle', 'ai/breaking', 'ai/pid'], outputs=['mux/steering', 'mux/throttle', 'mux/breaking'])
        boost_cfg = cfg['ai_boost']
        
        self.last_mode = DriveMode.HUMAN
        
        thr_lock_cfg = boost_cfg['launch_throttle_lock']
        self.throttle_lock_active = False
        self.throttle_lock_enabled = thr_lock_cfg['enabled']
        self.throttle_lock_value = thr_lock_cfg['value']
        self.throttle_lock_duration = thr_lock_cfg['duration']

        str_lock_cfg = boost_cfg['launch_steering_lock']
        self.steering_lock_active = False
        self.steering_lock_enabled = str_lock_cfg['enabled']
        self.steering_lock_value = str_lock_cfg['value']
        self.steering_lock_duration = str_lock_cfg['duration']

        self.pid = None



    def step(self, *args):
        toReturn = ()
        if args[0] == DriveMode.HUMAN:
            toReturn = args[1], args[2], args[3]
        elif args[0] == DriveMode.AI_STEERING:
            toReturn = args[4], args[2], args[3]
        elif args[0] == DriveMode.AI:
            toReturn = args[4], args[5], args[6]

        if self.last_mode != DriveMode.AI and args[0] == DriveMode.AI: #AI launch enable detection
            self.__start_throttle_lock()
            self.__start_steering_lock()
        
        if self.steering_lock_active:
            toReturn = self.steering_lock_value, toReturn[1], toReturn[2]
        if self.throttle_lock_active:
            toReturn = toReturn[0], self.throttle_lock_value, toReturn[2]

        self.last_mode = args[0]
        self.pid = args[7]
        #print(toReturn)
        return toReturn

    def getName(self):
        return 'Control Multiplexer'

    def __start_throttle_lock(self):
        if self.throttle_lock_enabled:
            print('[WARNING] Throttle Lock Activated')
            self.throttle_lock_active = True
            t = Thread(target=self.__end_throttle_lock, daemon=False)
            t.start()

    def __end_throttle_lock(self):
        time.sleep(self.throttle_lock_duration)
        self.throttle_lock_active = False
        print('[WARNING] Throttle Lock Ended')
        if self.pid is not None:
            self.pid.spd_pid.reset()
            print('Speed PID reset.')

    def __start_steering_lock(self):
        if self.steering_lock_enabled:
            self.steering_lock_active = True
            print('[WARNING] Steering Lock Activated')
            t = Thread(target=self.__end_steering_lock, daemon=False)
            t.start()

    def __end_steering_lock(self):
        time.sleep(self.steering_lock_duration)
        self.steering_lock_active = False
        print('[WARNING] Steering Lock Ended')
        if self.pid is not None:
            self.pid.str_pid.reset()
            print('Steering PID reset.')