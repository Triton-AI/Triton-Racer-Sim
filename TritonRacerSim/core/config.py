import json
import uuid
config = {
    'explanation': '''model_type: cnn_2d | cnn_2d_speed_as_feature | cnn_2d_speed_control | cnn_2d_full_house; joystick_type: ps4 | xbox | g28; sim_host: use 127.0.0.1 for local; track_data_file: used for position tracker to segment the track
    ''',

    'cam_w': 160,
    'cam_h': 120,

    'joystick_type': 'ps4',
    
    'ai_launch_boost_throttle_enabled': False,
    'ai_launch_boost_throttle_value': 1.0,
    'ai_launch_boost_throttle_duration': 5,

    'ai_launch_lock_steering_enabled': False,
    'ai_launch_lock_steering_value': 0.0,
    'ai_launch_lock_steering_duration': 3,

    'model_type': '2d_cnn',
    'early_stop': True,
    'early_stop_patience': 5,
    'max_epoch': 100,
    'speed_control_threshold': 1.1,
    'speed_control_reverse': 0.0,
    'speed_control_break': 0.0,

    'car_name': 'TritonRacer',
    'font_size': 50,
    'racer_name': 'Triton AI',
    'bio': 'Something',
    'country': 'US',
    'body_style': 'car01',
    'body_rgb': (24, 43, 73),
    'guid': 'will_be_overwritten_when_generating_config',

    'scene_name': 'mountain_track',
    'sim_path': 'remote',
    'sim_host': '127.0.0.1',
    'sim_port': 9091,
    'sim_latency': 0,
    
    'use_location_tracker': True,
    'track_data_file': 'centerline.json'

}

def read_config(config_path):
    with open(config_path, 'r') as config_file:
        cfg = json.load(config_file)
    return cfg

def generate_config(config_path):
    config['guid'] = uuid.uuid1().__str__()
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file, indent=4)