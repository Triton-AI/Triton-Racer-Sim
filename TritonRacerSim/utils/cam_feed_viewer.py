import cv2, socket, base64, time
import numpy as np
from io import BytesIO
from PIL import Image

if __name__ == "__main__":
    import cv2

    HOST = "localhost"
    PORT = 9094
    H = 180
    W = 320
    C = 3
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
                        # image = Image.open(BytesIO(base64.b64decode(msg)))
                        image = np.frombuffer(base64.b64decode(msg), dtype=np.uint8)
                        image = image.reshape((H, W, C))
                        cv2.imshow("Live Dashcam Feed", cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1) 
                        msg = data_string[idx:]
                    else:
                        msg += data_string
        except KeyboardInterrupt:
            pass