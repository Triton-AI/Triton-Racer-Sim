"""
Scripts to drive a triton racer car

Usage:
    manage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|latent)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>]
    manage.py (train) [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer)] [--continuous] [--aug] [--myconfig=<filename>]


Options:
    -h --help               Show this screen.
    --js                    Use physical joystick.
    -f --file=<file>        A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value>      Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
    --myconfig=filename     Specify myconfig file to use. 
                            [default: myconfig.py]
"""

import sys
sys.path.append('/home/haoru/Projects/Triton-Racer/Triton-Racer-Sim/')
from docopt import docopt
from TritonRacerSim.core.car import Car
from TritonRacerSim.core.datapool import DataPool


def assemble_car():
    car = Car(loop_hz=20)

    from TritonRacerSim.components.controller import PygameJoystick
    from TritonRacerSim.components.controlmultiplexer import ControlMultiplexer
    from TritonRacerSim.components.gyminterface import GymInterface
    from TritonRacerSim.components.datastorage import  DataStorage

    joystick = PygameJoystick(joystick_type='ps4')
    mux = ControlMultiplexer()
    gym = GymInterface(poll_socket_sleep_time=0.01)
    storage = DataStorage()

    car.addComponent(joystick)
    car.addComponent(mux)
    car.addComponent(gym)
    car.addComponent(storage)

    return car

if __name__ == '__main__':
    args = docopt(__doc__)
    if args['drive']:
        assemble_car().start()

