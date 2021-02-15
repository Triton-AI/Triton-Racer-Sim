# Stream the camera feed to another machine
# Use this same file on the client end.

import socket, base64, time
from PIL import Image
from io import BytesIO

from TritonRacerSim.components.component import Component

class CamFeedStreamer(Component):
    def __init__(self, cfg):
        Component.__init__(self, inputs=[cfg['image_topic']], threaded=True)
        self.frame = None
        self.on = True
        self.port = cfg['port']

    def onStart(self):
        """What should I do right at the launch of the car?"""
        pass

    def step(self, *args):
        self.frame = args[0]

    def thread_step(self):
        self.serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serv.bind(('0.0.0.0', self.port))
        self.serv.listen(1)
        while self.on:
            print("[Camera Feed Streamer] Awaiting Viewer Connection...")
            self.conn, addr = self.serv.accept()
            print(f"[Camera Feed Streamer] A viewer at {addr} is connected.")
            try:
                while self.on:
                    if self.frame is not None:
                        data = base64.b64encode(self.frame)
                        self.conn.sendall(bytes(data.decode('utf-8')+'\n', 'utf-8'))
                    time.sleep(0.02)
            except (ConnectionResetError, BrokenPipeError):
                print(f"[Camera Feed Streamer] Viewer at {addr} is disconnected.")
                continue
            finally:
                self.conn.close()



    def onShutdown(self):
        """What to do at shutdown?"""
        self.on = False

    def getName(self):
        return 'Camera Feed Streamer'