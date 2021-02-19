from TritonRacerSim.components.gyminterface import GymInterface
from PIL import Image

# Example code of wrapping a TritonRacer component to another platform.
'''
input: dictionary:
{
    'steering': 54,
    'throttle': 34,
    'breaking': 0.7,
    'reset': true
}

output: dictionary:
{
    'img': PIL image,
    'speed': 2444,
    'lidar': [01, 2, 304, 5903, ...]
}
'''

class gyminterface_wrapper:
    def __init__(self):
        self.gym = GymInterface()

    def interface(self, inputs):
        steering = inputs['steering']
        throttle = inputs['throttle']
        breaking = inputs['breaking']
        reset = inputs['reset']

        img, x, y, z, speed, cte, lidar = self.gym.step((steering, throttle, breaking, reset))

        outputs = {
                'img': Image.fromarray(img),
                'speed': speed,
                'lidar': lidar
        }
        return outputs


