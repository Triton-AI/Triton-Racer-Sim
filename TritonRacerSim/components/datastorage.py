from TritonRacerSim.components.component import Component
from pathlib import Path
from os import path
import os
from PIL import Image
import json
from threading import Thread
import sys

class DataStorage(Component):
    def __init__(self, to_store=['cam/img', 'mux/throttle', 'mux/steering', 'mux/break', 'gym/speed', 'pos/loc', 'usr/mode']):
        Component.__init__(self, inputs=to_store, threaded=False)
        self.step_inputs += ['usr/del_record', 'usr/toggle_record']

        self.storage_path = self.__getStoragePath()
        self.count = 0
        self.recording = False

    def step(self, *args):
        # print(args)
        #delete records
        if args[-2]:
            t = Thread(target=self.__delRecords, args=(100,),daemon=True)
            t.start()
        # store records
        elif args[-1]:          
            record = {self.step_inputs[i]: args[i] for i in range(len(self.step_inputs))}
            #Keep file IO in a seperate thread
            t = Thread(target=self.__store, args = (record,), daemon=True)
            t.start()

    def onShutdown(self):
        """Shutdown"""
        pass

    def getName(self):
        return 'Data Storage'
        
    def __getStoragePath(self):
        car_path = sys.path[0]
        data_dir = path.join(car_path, 'data/')

        i = 1
        while path.exists(path.join(data_dir, f'records_{i}/')):
            i += 1
        
        dir_path = path.join(data_dir, f'records_{i}/')
        os.mkdir(dir_path)
        return dir_path

    def __store(self, record={}):
        self.__storeImg(record)

        record_path = path.join(self.storage_path, f'record_{self.count}.json')
        with open(record_path, 'w') as recordFile:
            json.dump(record, recordFile)
        
        self.count += 1


    def __storeImg(self, record={}):
        if record['cam/img'] is not None:
            img_path = path.join(self.storage_path, f'img_{self.count}.jpg')
            record['cam/img'].save(img_path)
            record['cam/img'] = f'img_{self.count}.jpg'

    def __delRecords(self, num):
        original_count = self.count
        self.count -= num
        if (self.count < 0) self.count = 0
        for i in range(self.count, original_count):
            img_path = path.join(self.storage_path, f'img_{i}.jpg')
            record_path = path.join(self.storage_path, f'record_{i}.json')
            os.remove(img_path)
            os.remove(record_path)