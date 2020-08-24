"""
Scripts to drive a triton racer car

Usage:
    manage.py (drive) [--model=<model>] [--js]
    manage.py (train) (--tub=<tub1,tub2,..tubn>) (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer)]


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
from os import path
from TritonRacerSim.core.car import Car
from TritonRacerSim.core.datapool import DataPool
from TritonRacerSim.utils.types import ModelType

def assemble_car(model_path = None, model_type = None):
    car = Car(loop_hz=20)

    from TritonRacerSim.components.controller import PygameJoystick, G28DrivingWheel, PS4
    from TritonRacerSim.components.controlmultiplexer import ControlMultiplexer
    from TritonRacerSim.components.gyminterface import GymInterface
    from TritonRacerSim.components.datastorage import  DataStorage

    #joystick = G28DrivingWheel()
    joystick = PS4()
    mux = ControlMultiplexer()
    gym = GymInterface(poll_socket_sleep_time=0.01)
    storage = DataStorage()

    if model_path is not None:
        from TritonRacerSim.components.keras_pilot import KerasPilot
        pilot = KerasPilot(model_path, model_type)
        car.addComponent(pilot)

    car.addComponent(joystick)
    car.addComponent(mux)
    car.addComponent(gym)
    car.addComponent(storage)

    return car

if __name__ == '__main__':
    args = docopt(__doc__)
    if args['drive']:
        model_path =None
        if args['--model']:
            model_path = path.abspath(args['--model'])

        assemble_car(model_path, ModelType.CNN_2D).start()

    if args['train']:
        tub = args['--tub']
        data_paths = []
        for folder_path in tub.split(','):
            data_paths.append(path.abspath(folder_path))

        model_path = path.abspath(args['--model'])
        # assert path.exists(model_path)

        from TritonRacerSim.components.keras_train import train
        train(model_type=ModelType.CNN_2D, img_shape = (180, 320, 3), data_paths=data_paths, model_path=model_path)

