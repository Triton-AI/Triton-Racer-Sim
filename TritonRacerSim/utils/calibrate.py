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
    elif board == 'VESC':
        calibrate_vesc(cfg['vesc'], args)
    else: raise Exception(f'Unknown board type {board}.')   

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

def calibrate_vesc(cfg, args):
    from TritonRacerSim.components.vesc import VESC_
    v = VESC_(cfg)
    try:
        while True:
            if args['--steering']:
                v.send_servo_angle(ask_for_angle())
            elif args['--rpm']:
                v.send_rpm(ask_for_rpm())
            elif args['--throttle']:
                v.send_duty_cycle(ask_for_duty_cycle())
    except KeyboardInterrupt:
        v.onShutdown()
        return
    

def ask_for_pwm():
    return int(input("Enter a PWM ([0, 4095], < 500 recommanded): "))

def ask_for_rpm():
    return int(input("Enter an RPM ([-30000, 30000], [-10000, 20000] recommanded, 0 is neutral): "))

def ask_for_angle():
    while True:
        angle = float(input("Enter a steering angle ([0, 1]): "))
        if 0 <= angle <= 1: return angle
        else:
            print("Angle out of range.")

def ask_for_duty_cycle():
    return int(input("Enter a Duty Cycle ([-1e5, 1e5], [-10000, 20000] recommanded, 0 is neutral): "))