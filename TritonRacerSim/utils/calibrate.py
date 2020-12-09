import json

def calibrate(cfg, args):
    board = cfg['sub_board_type']
    print(f'Calibrating {board}')
    if board == 'PCA9685':
        calibrate_pca9685(cfg['PCA9685'], args)
    elif board == 'TEENSY':
        calibrate_teensy(cfg['teensy'], args)
    elif board == 'ESP32':
        calibrate_esp32(cfg, args)
    else: raise Exception(f'Unknown board type {board}')   

def calibrate_pca9685(cfg, args):
    pass

def calibrate_teensy(cfg, args):
    import serial
    whatToCalibrate = ''
    if args['--steering']:
        whatToCalibrate = 'Steering'
    elif args['--throttle']:
        whatToCalibrate = 'Throttle'
    else: raise Exception(f'Please specify which control to calibrate (python manage.py calibrate steering / throttle).')

    ser = serial.Serial(port=cfg['port'], baudrate=cfg['baudrate'])

    while True:
        pwm = ask_for_pwm()
        msg = f"try{whatToCalibrate}_{pwm}\n"
        ser.write(bytes(msg, 'utf-8'))

def calibrate_esp32(cfg, args):
    from TritonRacerSim.components.esp32_cam import ESP32CAM
    esp = ESP32CAM(cfg)
    steering = esp.neutral_steering_pulse
    throttle = esp.zero_pulse
    while True:
        pwm = ask_for_pwm()
        if args['--steering']:
            steering = pwm
        elif args['--throttle']:
            throttle = pwm
        msg_dict = {'msg_type': 'control', 'steering': steering, 'throttle': throttle}
        msg = json.dumps(msg_dict) + '\n'
        esp.send(msg)

def ask_for_pwm():
    return int(input("Enter a PWM (0-4095, < 500 recommanded): "))

