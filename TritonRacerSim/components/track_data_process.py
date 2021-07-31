from os import path
import json
import time
from PIL import Image
from numpy.core.numeric import indices

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
                point = {'gym/x': data['gym/x'], 'gym/z': data['gym/z'], 'yaw': data['gym/telemetry']['yaw']}
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
    def __init__(self, cfg):
        Component.__init__(self, inputs=['gym/x', 'gym/y', 'gym/z', 'gym/telemetry'], outputs=['loc/segment', 'loc/cte', 'loc/break_indicator', 'loc/cte_pred', 'loc/heading', 'loc/heading_pred'])
        seg_data_path = cfg['seg_data_file']
        cte_data_path = cfg['cte_data_file']

        with open(seg_data_path, 'r') as input_file:
            self.seg_data = json.load(input_file)
        with open(cte_data_path, 'r') as input_file:
            self.cte_data = json.load(input_file)

        seg_coords = [(pt['gym/x'], pt['gym/z']) for pt in self.seg_data]
        cte_coords = [(pt['gym/x'], pt['gym/z']) for pt in self.cte_data]
        self.cte_headings = [pt['yaw'] for pt in self.cte_data]

        self.seg_path = geom.LineString(seg_coords)
        self.cte_path = geom.LineString(cte_coords)
        self.cte_ring = geom.LinearRing(cte_coords)
        self.cte_poly = geom.Polygon(cte_coords)

        self.break_region = cfg['break_region']

    def localize(self, point):
        pt = geom.Point(*point)
        cte = pt.distance(self.cte_ring)
        if not self.cte_poly.contains(pt):
            cte *= -1.0
        proj = self.seg_path.project(pt, normalized=True)
        return proj, cte
        

    def step(self, *args):
        x, y, z, tele = args
        track_segment, cte = self.localize((x, z))
        coord_pred = self.__position_approximation(tele)
        pred_segment, pred_cte = self.localize(coord_pred)
        # loc = ["{0:0.2f}".format(p) for p in args]
        # seg_str = "{0:0.4f}".format(track_segment)
        # cte_str = "{0:0.4f}".format(cte)
        # print(f'Segment: {seg_str}, CTE: {cte_str}\r', end='')
        indicator = self.__calc_break_indicator(track_segment)
        cte_heading = self.__calc_heading((x, z))
        cte_heading_pred = self.__calc_heading(coord_pred)

        return track_segment, cte, indicator, pred_cte, cte_heading, cte_heading_pred

    def getName(self):
        return "Location Tracker"

    def __in_region(self, segment, region):
        b1, b2 = region
        return (b1 <= segment <= b2) or (b2 <= segment <= b1)

    def __calc_break_indicator(self, segment):
        if self.break_region is not None and None not in self.break_region:
            for region in self.break_region:
                if self.__in_region(segment, region):
                    return 1
        return 0

    def __position_approximation(self, tele):
        # Second order position approximation using velosity and acceleration

        dt = 0.05
        
        x = tele.pos_x
        z = tele.pos_z
        accel_x = tele.accel_x
        accel_z = tele.accel_y
        vel_x = tele.vel_x
        vel_z = tele.vel_z

        x_pred = x + vel_x * dt + 0.5 * accel_x * dt ** 2
        z_pred = z + vel_z * dt + 0.5 * accel_z * dt ** 2

        return x_pred, z_pred

    def __calc_heading(self, pt):
        pt = geom.Point(*pt)
        proj = self.cte_path.project(pt, normalized=True)
        return self.cte_headings[int((len(self.cte_headings)-1)*proj )]
