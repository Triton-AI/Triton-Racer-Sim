import socket, base64, time, pickle
import numpy as np
from io import BytesIO
from threading import Thread
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


class LidarReceiver:
    def __init__(self):
        self.lidar = None
        self.t = Thread(target=self.recv, daemon=False)
        self.t.start()

    def animate(self, i):
        # print('here')
        if self.lidar is not None:
            # print('hereb')
            degs, norms = self.lidar
            a = plt.axes()
            a.set
            plt.cla()
            plt.polar(degs, norms)
            plt.title("DonkeyGym LiDAR")
            

    def depackage(self, lidar):
        x = [scan['x'] for scan in lidar]
        z = [scan['z'] for scan in lidar]
        return x, z


    def recv(self):
        HOST = "localhost"
        PORT = 9094

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            f = s.makefile()
            try: 
                while True:
                    msg = ""
                    while True:
                        data = s.recv(1024)
                        data_string = data.decode('utf-8')
                        if '\n' in data_string:
                            idx = data_string.index('\n')
                            msg += data_string[0:idx]
                            self.lidar = pickle.loads(base64.b64decode(msg))
                            msg = data_string[idx:]
                        else:
                            msg += data_string
            except KeyboardInterrupt:
                pass

if __name__ == "__main__":
    viewer = LidarReceiver()
    ani = FuncAnimation(plt.gcf(), viewer.animate, interval=100)
    plt.show()