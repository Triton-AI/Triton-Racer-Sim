import json
import uuid
config = {
    'explanation': '''model_type: cnn_2d | cnn_2d_speed_as_feature | cnn_2d_speed_control | cnn_2d_full_house; joystick_type: ps4 | xbox | g28; sim_host: use 127.0.0.1 for local; track_data_file: used for position tracker to segment the track
    ''',

    'img_w': 160,
    'img_h': 120,

    'preprocessing_enabled': False, # Enable image filtering. Original image is NOT preserved at this stage.
    'preprocessing_contrast_enhancement_ratio': 1.0, # Enhance contrast, especially on the new robo racing league track
    'preprocessing_contrast_enhancement_offset': 125, # Ranging [0,255]. Pixels above this value will be positively boosted, vise versa, and hence enhancing the contrast of the image.
    'preprocessing_color_filter_enabled': False, # Filtering out the interested colors. Checkout Triton AI Color Filter Tutorial.
    'preprocessing_color_filter_hsvs': [((0, 0, 130),(180, 64, 255)),((25, 180, 155),(43, 255, 255))], # upper and lower bounds (HSV) of each color detection. In this case white and yellow.
    'preprocessing_color_filter_destination_channels':[0, 1], # Which channel to put the filtered layers? 0 | 1 | 2 for RGB image. Must match the number of hsv filters above.)
    'preprocessing_edge_detection_enabled': False, # Apply a canny filter for edge detection
    'preprocessing_edge_detection_threshold_a': 60, # Threshold used in OpenCV canny filter
    'preprocessing_edge_detection_threshold_b': 100,
    'preprocessing_edge_detection_destination_channel': 2, # Which channel to put the filtered layer? 0 | 1 | 2 for RGB image

    'joystick_type': 'ps4', # ps4 | xbox | g28 Wired joysticks recommended. ps4 joystick over bluetooth may end up with different joystick mappings. WIP.
    
    'ai_launch_boost_throttle_enabled': False, # Lock throttle when switching from ai-steering to full-ai mode
    'ai_launch_boost_throttle_value': 1.0,
    'ai_launch_boost_throttle_duration': 5,

    'ai_launch_lock_steering_enabled': False, # Lock steering when switching from ai-steering to full-ai mode
    'ai_launch_lock_steering_value': 0.0,
    'ai_launch_lock_steering_duration': 3,

    'smooth_steering_enabled': True, # Consider all AI steerings above the threshold a full steering (1.0 or -1.0)
    'smooth_steering_threshold': 0.9,

    'model_type': 'cnn_2d_speed_control', # cnn_2d | cnn_2d_speed_as_feature | cnn_2d_speed_control | cnn_2d_full_house
    'early_stop': True, # Early stop when training hasn't made any progress within the patience
    'early_stop_patience': 5,
    'max_epoch': 100, # Max epoch to train
    'speed_control_threshold': 1.1, # Allow the model to overspeed. 1.1 means 10% speeding.
    'speed_control_reverse': 0.0, # Apply reverse throttle when overspeed, e.g. -0.4.
    'speed_control_break': 0.0, # Apply break when overspeed, e.g. 0.3. Break will OVERRIDE any throttle value.
    'batch_size': 64, # Lower it to save GPU resources, or increase it to experdite training.

    'car_name': 'TritonRacer',
    'font_size': 50,
    'racer_name': 'Triton AI',
    'bio': 'Something',
    'country': 'US',
    'body_style': 'car01',
    'body_rgb': (24, 43, 73),
    'guid': 'will_be_overwritten_when_generating_config',

    'scene_name': 'roboracingleague_1',
    'sim_path': 'remote', # Does not work currently
    'sim_host': '127.0.0.1',
    'sim_port': 9091,
    'sim_latency': 0,
    
    'use_location_tracker': False, # Only for mountain track currently. Disable it when on other tracks.
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