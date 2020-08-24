from TritonRacerSim.components.component import Component
from TritonRacerSim.components.controller import DriveMode

class ControlMultiplexer(Component):
    '''Switch user or ai control based on mode'''
    def __init__(self):
        Component.__init__(self, inputs=['usr/mode', 'usr/steering', 'usr/throttle', 'usr/breaking', 'ai/steering', 'ai/throttle', 'ai/breaking'], outputs=['mux/steering', 'mux/throttle', 'mux/breaking'])

    def step(self, *args):
        if args[0] == DriveMode.HUMAN:
            return args[1], args[2], args[3]
        elif args[0] == DriveMode.AI_STEERING:
            return args[4], args[2], args[3]
        elif args[0] == DriveMode.AI:
            # print('here')
            # print(args)
            return args[4], args[5], args[6]

    def getName(self):
        return 'Control Multiplexer'