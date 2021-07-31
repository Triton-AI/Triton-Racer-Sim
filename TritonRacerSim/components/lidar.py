from TritonRacerSim.components.component import Component
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.animation import FuncAnimation
import socket, base64, pickle
class DonkeySimLiDAR(Component):
    """LiDAR support for donkeygym"""
    def __init__(self, cfg):
        super().__init__(inputs=['gym/lidar'], outputs=['lidar/degs', 'lidar/distances', 'lidar/valid_degs', 'lidar/valid_distances'], threaded=False)
        self.on = True
        self.deg_inc = cfg['deg_inc']
        self.max_range = cfg['max_range']
        self.num_sweeps_levels = cfg['num_sweeps_levels']

        self.x = None
        self.y = None
        self.z = None
        self.degs = None
        self.norms = None
        # self.ani = FuncAnimation(plt.gcf(), self.animate, interval=100)
        # plt.show(block=False)


    def step(self, *args):
        self.lidar = args[0]
        return self.depackage(self.lidar)
    '''
    def draw(self):
        ani = FuncAnimation(plt.gcf(), self.animate, interval=100)
        plt.show()
        pass
    '''
    
    def depackage(self, lidar):
        # return None, None, None, None
        if lidar is None:
            return None, None, None, None
        dists = np.full(int(360 / self.deg_inc), self.max_range)
        degs = np.linspace(0, 360, num=int(360/self.deg_inc), endpoint=False)
        valid_dists, valid_degs = list(map(list, zip(*[(scan['d'], scan['rx']) for scan in lidar])))
        valid_idx = np.in1d(degs, valid_degs).nonzero()[0]
        dists[valid_idx] = valid_dists
        return degs, dists, valid_degs, valid_dists
        
    '''
    def animate(self, i):
        print('here')
        if self.lidar is not None:
            print('hereb')
            x, z = self.depackage(self.lidar)
            plt.cla()
            plt.scatter(x, z)
            plt.title("DonkeyGym LiDAR")
            plt.tight_layout()
    '''
    '''
    def get_polar(self, lidar, x, y, z):
        lidar_arr = np.asarray(self.depackage(lidar)).T
        norms = np.linalg.norm(lidar_arr - np.array([x, z]), axis=1).tolist()
        degs = np.linspace(0, 2 * lidar_arr.shape[0] - 2, num = lidar_arr.shape[0]).tolist()
        #print (degs)
        #print(norms)
        return degs, norms
    '''
    '''
    def thread_step(self):
        self.serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serv.bind(('0.0.0.0', self.port))
        self.serv.listen(1)
        while self.on:
            print("[LiDAR streamer] Awaiting Viewer Connection...")
            self.conn, addr = self.serv.accept()
            print(f"[LiDAR streamer] A viewer at {addr} is connected.")
            try:
                while self.on:
                    if self.lidar is not None:
                        data = base64.b64encode(pickle.dumps((self.degs, self.norms)))
                        self.conn.sendall(bytes(data.decode('utf-8')+'\n', 'utf-8'))
                    time.sleep(0.02)
            except (ConnectionResetError, BrokenPipeError):
                print(f"[LiDAR Streamer] Viewer at {addr} is disconnected.")
                continue
            finally:
                self.conn.close()
    '''
    '''
    def update_visualize(self, x, z):
        xdata = np.append(self.vis.get_xdata(), x)
        zdata = np.append(self.vis.get_ydata(), z)
        max_data = int(360 / self.deg_inc)
        if max_data < len(xdata):
            xdata = xdata[-max_data:]
            zdata = zdata[-max_data:]

        self.vis.set_xdata(xdata)
        self.vis.set_ydata(zdata)
        plt.pause(0.001)
        plt.show()
    '''

    def onShutdown(self):
        self.on = False

    def getName(self):
        return 'DonkeyGym LiDAR'