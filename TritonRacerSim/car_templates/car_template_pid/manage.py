"""
Scripts to drive a triton racer car

Usage:
    manage.py (drive) [--dummy]
    manage.py (generateconfig)
    manage.py (calibrate) [--steering] [--throttle] [--rpm]
    manage.py (processtrack) (--tub=<data_folder>) (--output=<track_json_file>)
    manage.py (joystick) [--dump]
"""

import sys
sys.path.append('/home/haoru/Projects/TR/Triton-Racer-Sim/')
from docopt import docopt
from os import path
import time
from TritonRacerSim.core.car import Car
from TritonRacerSim.utils.types import ModelType

def get_joystick_by_name(joystick_name, js_cfg):
    from TritonRacerSim.components.controller import JoystickType, G28DrivingWheel, PS4Joystick, XBOXJoystick, SWITCHJoystick, F710Joystick, CustomJoystick
    joystick_type = JoystickType(joystick_name)
    if joystick_type == JoystickType.PS4:
        return PS4Joystick(js_cfg)
    elif joystick_type == JoystickType.G28:
        return G28DrivingWheel(js_cfg)
    elif joystick_type == JoystickType.XBOX:
        return XBOXJoystick(js_cfg)
    elif joystick_type == JoystickType.SWITCH:
        return  SWITCHJoystick(js_cfg)
    elif joystick_type == JoystickType.F710:
        return F710Joystick(js_cfg)
    elif joystick_type == JoystickType.CUSTOM:
        return CustomJoystick(js_cfg)
    else:
        raise Exception(f'Unsupported joystick type: {joystick_type}. Could be still under development.')

def assemble_car(cfg = {}, args = {}, model_path = None):
    car = Car(loop_hz=cfg['loop_hz'])

    from TritonRacerSim.components.controlmultiplexer import ControlMultiplexer
    from TritonRacerSim.components.datastorage import  DataStorage
    from TritonRacerSim.components.driver_assistance import DriverAssistance

    # Verbose
    if len(cfg['verbose']) > 0:
        from TritonRacerSim.components.verbose import Verbose
        vb = Verbose(cfg)
        car.addComponent(vb)

    # PID Pilot
    if cfg['ai_model']['model_type'] == 'waypoint_follower':
        from TritonRacerSim.components.pid_pilot import WaypointFollower
        pilot = WaypointFollower(cfg['pid_pilot'], cfg['location_tracker'])
        car.addComponent(pilot)

    # Joystick
    js_cfg = cfg['joystick']
    if not args['--dummy']:
        joystick = get_joystick_by_name(js_cfg['type'], js_cfg)
        car.addComponent(joystick)
    else:
        from TritonRacerSim.components.controller import DummyJoystick
        joystick = DummyJoystick(js_cfg)
        car.addComponent(joystick)

    # Control Signal Multiplexer
    mux = ControlMultiplexer(cfg)
    car.addComponent(mux)

    # Driver Assistance
    drive_assist_cfg = cfg['drive_assist']
    if drive_assist_cfg['enabled']:
        assistant = DriverAssistance(drive_assist_cfg)
        car.addComponent(assistant)

    # Interface with donkeygym, or real car electronics
    if cfg['I_am_on_simulator']:
        from TritonRacerSim.components.gyminterface import GymInterface
        gym = GymInterface(poll_socket_sleep_time=0.01,gym_config=cfg['simulator'])
        car.addComponent(gym)
    else:
        sub_board = cfg['electronics']['sub_board_type']
        if sub_board == 'PCA9685':
            raise NotImplementedError()
        if sub_board == 'TEENSY':
            from TritonRacerSim.components.teensy import TeensyMC_Test
            teensy = TeensyMC_Test(cfg['electronics'])
            car.addComponent(teensy)
        elif sub_board == 'ESP32':
            from TritonRacerSim.components.esp32_cam import ESP32CAM
            esp = ESP32CAM(cfg['electronics'])
            car.addComponent(esp)
        elif sub_board == 'VESC':
            from TritonRacerSim.components.vesc import VESC_
            v = VESC_(cfg['electronics']['vesc'])
            car.addComponent(v)
        else:
            raise Exception(f"Sub-board type not recognized: {sub_board}.")

        cam_cfg = cfg['cam']
        if cam_cfg['type'] == 'WEBCAM' and sub_board != 'ESP32':
            from TritonRacerSim.components.camera import Camera
            cam = Camera(cam_cfg)
            car.addComponent(cam)
        elif cam_cfg['type'] == 'WEBCAM_OPENCV' and sub_board != 'ESP32':
            from TritonRacerSim.components.camera import OpenCVCamera
            cam = OpenCVCamera(cam_cfg)
            car.addComponent(cam)

    lidar_cfg = cfg['simulator']['lidar']
    if cfg['I_am_on_simulator'] and lidar_cfg['enabled']:
        from TritonRacerSim.components.lidar import DonkeySimLiDAR
        lidar = DonkeySimLiDAR(lidar_cfg)
        car.addComponent(lidar)

    #Image preprocessing (Open CV Required)
    prep_cfg = cfg['img_preprocessing']
    if prep_cfg['enabled']:
        from TritonRacerSim.components.img_preprocessing import ImgPreprocessing
        preprocessing = ImgPreprocessing(prep_cfg)
        car.addComponent(preprocessing)

    # Location tracker
    loc_cfg = cfg['location_tracker']
    if loc_cfg['enabled']:
        from TritonRacerSim.components.track_data_process import LocationTracker
        tracker = LocationTracker(loc_cfg)
        car.addComponent(tracker)
        
    cam_cfg = cfg['cam']
    # Final Image Scaler (Open CV Required)
    if (cam_cfg['img_h'], cam_cfg['img_w']) != cam_cfg['native_resolution'] and cfg['I_am_on_simulator']:
        from TritonRacerSim.components.img_preprocessing import ImageResizer
        resizer = ImageResizer(cam_cfg)
        car.addComponent(resizer)

    # Data storage
    storage = DataStorage()
    if prep_cfg['enabled']:
        storage.step_inputs[0] = 'cam/processed_img'
        if prep_cfg['keep_original']:
            original_data_storage = DataStorage(storage_path=storage.storage_path[0:-1]+'_original/')
            car.addComponent(original_data_storage)
    car.addComponent(storage)

    # Image Feed Streamer
    img_stm = cfg['stream_cam_feed']
    if img_stm['enabled']:
        from TritonRacerSim.components.cam_feed_streamer import CamFeedStreamer
        streamer = CamFeedStreamer(img_stm)
        car.addComponent(streamer)

    return car

if __name__ == '__main__':
    args = docopt(__doc__)
    if args['generateconfig']:
        #from TritonRacerSim.core.config import generate_config
        #generate_config('./myconfig.json')
        print('\"manage.py generateconfig\" has been depreciated! You will find a \"myconfig.yaml\" in your car folder instead.')

    elif args['processtrack']:
        from TritonRacerSim.components.track_data_process import TrackDataProcessor
        processor = TrackDataProcessor(args['--tub'], args['--output'])
        processor.process()

    else:
        from TritonRacerSim.core.config import read_config, read_yaml_config
        #cfg = read_config(path.abspath('./myconfig.json'))
        cfg = read_yaml_config(path.abspath('./myconfig.yaml'))
        if args['drive']:
            sim_launch = cfg['simulator_autolaunch']
            sim_path = sim_launch['donkey_sim_full_path']
            if sim_path != 'remote' and cfg['I_am_on_simulator'] and cfg['simulator']['default_connection'] == 'local':
                import subprocess, time
                print (f'[Launching Local Simulator] {sim_path}')
                subprocess.Popen(sim_path)
                time.sleep(5)
            assemble_car(cfg, args).start()

        elif args['calibrate']:
            from TritonRacerSim.utils.calibrate import calibrate
            calibrate(cfg['electronics'], args)

        elif args['joystick']:
            from TritonRacerSim.components.controller import CustomJoystickCreator
            creator = CustomJoystickCreator()
            if args['--dump']:
                creator.js = creator.select_joystick()
                while True:
                    print(f"{creator.dump_joystick(creator.js)}\r",end="")
                    time.sleep(0.5)
            else:
                creator.create()