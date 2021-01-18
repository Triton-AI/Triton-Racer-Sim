from os import path
import json
import time
from PIL import Image

import shapely.geometry as geom

from TritonRacerSim.components.component import Component


class TrackDataProcessor:
    def __init__(self, tub_path, output_path):
        self.tub_path = tub_path
        self.output_path = output_path
        if not path.exists(tub_path):
            raise FileNotFoundError('Cannot find tub {}'.format(tub_path))
        self.line = []

    def process(self):
        i = 1
        while True:
            try:
                data_path = path.join(self.tub_path, 'record_{}.json'.format(i))

                f = open(data_path)
                data = json.load(f)
                f.close()

                # point = [data['gym/x'], data['gym/y'], data['gym/z']]
                point = [data['gym/x'], data['gym/z']]
                self.line.append(point)

                i += 1
            except FileNotFoundError:
                break

        print(i, 'points loaded, Saving to ', self.output_path)

        # self.__sort()

        with open(self.output_path, 'w') as output_file:
            json.dump(self.line, output_file)

    def __sort(self):
        '''buggy'''
        original_count = len(self.line)
        newData = []
        last_point = self.line[0]
        selected_i = 0
        
        while len(newData) != original_count:          
            last_distance = 100
            for i, point in enumerate(self.line):
                current_distance = self.__distance(last_point, point)
                if current_distance < last_distance:
                    selected_i = i
                    last_distance = current_distance

            newData.append(self.line[selected_i])
            last_point = newData[-1]
            del self.line[selected_i]
        
        self.line = newData


    def __distance(self, a, b):
        return abs((a[0] - b[0]) + (a[0] - b[0]) + (a[0] - b[0]))



class LocationTracker(Component):
    def __init__(self, seg_data_path, cte_data_path, min_map = 0, max_map = 10):
        Component.__init__(self, inputs=['gym/x', 'gym/y', 'gym/z'], outputs=['loc/segment', 'loc/cte'])

        with open(seg_data_path, 'r') as input_file:
            self.seg_data = json.load(input_file)
        with open(cte_data_path, 'r') as input_file:
            self.cte_data = json.load(input_file)
        self.max = max_map
        self.min = min_map
        self.seg_path = geom.LineString(self.seg_data)
        self.cte_ring = geom.LinearRing(self.cte_data)
        self.cte_poly = geom.Polygon(self.cte_data)

    def localize(self, point):
        pt = geom.Point(*point)
        cte = pt.distance(self.cte_ring)
        if not self.cte_poly.contains(pt):
            cte *= -1.0
        proj = self.seg_path.project(pt, normalized=True)
        return proj, cte
        

    def step(self, *args):
        track_segment, cte = self.localize((args[0], args[2]))
        loc = ["{0:0.2f}".format(p) for p in args]
        seg_str = "{0:0.4f}".format(track_segment)
        cte_str = "{0:0.4f}".format(cte)
        # print(f'Segment: {seg_str}, CTE: {cte_str}\r', end='')
        return track_segment, cte

    def onShutdown(self):
        pass

    # obsolete
    def __find_closest(self, point):
        begin_time = time.time()

        selected_i = 0
        last_distance = 100

        for i, center_point in enumerate(self.data):
            current_distance = self.__distance(point, center_point)
            if current_distance < last_distance:
                    selected_i = i
                    last_distance = current_distance
        duration = time.time() - begin_time
        return selected_i, duration

    # obsolete
    def __distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])    

    # obsolete
    def __map(self, idx):
        return idx / float(len(self.data)) * (self.max - self.min) + self.min
            

