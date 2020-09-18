'''If you have the original images, and you would like to filter the image based on new filter params, use post_processing.'''
import os
import time
from os import path
from shutil import copyfile
from threading import Thread
import json

from TritonRacerSim.components.img_preprocessing import ImgPreprocessing
from TritonRacerSim.components.datastorage import DataStorage

import cv2

def post_process(source, destination, cfg={}):
    print("[Post-processing]")
    print(f'Source: {source}')
    print(f'Destination: {destination}')

    processor = ImgPreprocessing(cfg)
    t = Thread(target=processor.step_inputs, daemon=True)
    t.start()

    source = path.abspath(source)
    destination = path.abspath(destination)
    os.mkdir(destination)

    count = 0
    try: 
        while True:
            img = cv2.cvtColor(cv2.imread(path.join(source, f'img_{count}.jpg')), cv2.COLOR_BGR2RGB)
            processor.step(img,)
            copyfile(path.join(source, f'record_{count}.json'), path.join(destination, f'record_{count}.json'))
            while processor.processed_img is None:
                time.sleep(0.005)
            cv2.imwrite(path.join(destination, f'img_{count}.jpg'))
            processor.processed_img = None
            count += 1

    except FileNotFoundError:
        print(f'{count} records processed.')