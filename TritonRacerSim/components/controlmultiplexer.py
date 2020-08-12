from TritonRacerSim.components.component import Component
from TritonRacerSim.components.controller import DriveMode

class ControlMultiplexer(Component):
    '''Switch user or ai control based on mode'''
    def __init__(self):
        Component.__init__(self, inputs=['usr/mode', 'usr/steering', 'usr/throttle', 'ai/steering', 'ai/throttle'], outputs=['mux/steering', 'mux/throttle'])

    def step(self, *args):
        if args[0] == str(DriveMode.HUMAN):
            return args[1], args[2]
        elif args[0] == str(DriveMode.AI_STEERING):
            return args[3], args[2]
        elif args[0] == str(DriveMode.AI):
            return args[3], args[4]

    def getName(self):
        return 'Control Multiplexer'