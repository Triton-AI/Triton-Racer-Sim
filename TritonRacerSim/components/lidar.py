from TritonRacerSim.components.component import Component
import matplotlib.pyplot as plt
import numpy as np
class DonkeySimLiDAR(Component):
    """HACK brocken"""
    """LiDAR support for donkeygym"""
    def __init__(self, cfg):
        super().__init__(inputs=['gym/lidar'], outputs=[], threaded=True)
        self.on = True
        self.deg_inc = cfg['deg_inc']
        self.max_range = cfg['max_range']

    def onStart(self):
        #plt.ion()
        #self.vis, = plt.plot([],[])
        #plt.show()
        pass

    def step(self, *args):
        #lidar, = args
        #if lidar is not None:
        #    x, z = self.depackage(lidar)
        #    self.update_visualize(x, z)
        pass
    
    def depackage(self, lidar):
        x = [scan['x'] for scan in lidar]
        z = [scan['z'] for scan in lidar]
        return x, z

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

    def thread_step(self):
        """The component's behavior in its own thread"""
        pass

    def onShutdown(self):
        self.on = False
        plt.ioff()

    def getName(self):
        return 'DonkeyGym LiDAR'