import json

class TelemetryPack:
    def __init__(self, steering=0.0, throttle=0.0, speed=0.0, 
                pos_x=None, pos_y=None, pos_z=None, hit=None, 
                time=0.0, accel_x=None, accel_y=None, accel_z=None, 
                gyro_x=None, gyro_y=None, gyro_z=None, gyro_w=None, 
                pitch=None, yaw=None, roll=None, cte=None, 
                active_node=None, total_nodes = None, vel_x=None, 
                vel_y=None, vel_z=None, on_road=None, 
                progress_on_track=None):
        self.steering = steering
        self.throttle = throttle
        self.speed = speed
        self.pos_x = pos_x
        self.pos_y= pos_y
        self.pos_z = pos_z
        self.hit = hit
        self.time = time
        self.accel_x = accel_x
        self.accel_y = accel_y
        self.accel_z = accel_z
        self.gyro_x = gyro_x
        self.gyro_y = gyro_y
        self.gyro_z = gyro_z
        self.gyro_w = gyro_w
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.cte = cte
        self.active_node = active_node
        self.total_nodes = total_nodes
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_z = vel_z
        self.on_road = on_road
        self.progress_on_track = progress_on_track
    
    def __str__(self) -> str:
        return json.dumps(self.__dict__)

class ObjectPack:
    def __init__(self, x, y, z, cte, in_track) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.cte = cte
        self.in_track = in_track