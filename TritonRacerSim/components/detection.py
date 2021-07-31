from numpy.lib.function_base import append
from TritonRacerSim.components.component import Component
import shapely.geometry as geom
import json
from TritonRacerSim.utils.telemetry import ObjectPack
import numpy as np

'''
class DonkeyGymClustering(Component):
    def __init__(self, cfg):
        super().__init__(inputs=['lidar/valid_degs', 'lidar/valid_distances'], outputs=['detection/clusterings'], threaded=False)
        seg_data_path = cfg['seg_data_file']
        cte_data_path = cfg['cte_data_file']
        left_data_path = cfg['left_data_file']
        right_data_path= cfg['right_data_file']
        with open(seg_data_path, 'r') as input_file:
            self.seg_data = json.load(input_file)
        with open(cte_data_path, 'r') as input_file:
            self.cte_data = json.load(input_file)
        with open(left_data_path, 'r') as input_file:
            self.left_data = json.load(input_file)
        with open(right_data_path, 'r') as input_file:
            self.right_data = json.load(input_file)
        self.seg_path = geom.LineString(self.seg_data)
        self.cte_ring = geom.LinearRing(self.cte_data)
        self.cte_poly = geom.Polygon(self.cte_data)
        self.left_poly = geom.Polygon(self.left_data)
        self.right_poly = geom.Polygon(self.right_data)


    def localize(self, point):
        pt = geom.Point(*point)
        cte = pt.distance(self.cte_ring)
        if not self.cte_poly.contains(pt):
            cte *= -1.0
        proj = self.seg_path.project(pt, normalized=True)
        return proj, cte
        

    def step(self, *args):
        lidar_degs, lidar_dists = args
        coordinates = self.__cluster(lidar_degs, lidar_dists)
        return self.__check_if_on_track(coordinates)

    def __polar_distance(self, r1, r2, theta1, theta2):
        return np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(theta2 - theta1))

    def __cluster(self, lidar_degs, lidar_distances):
        gap_threshold = 2
        object_coords = []
        object_deg_ranges = []
        begin = 0
        end = 0
        for i in range(len(lidar_distances)):
            if (self.__polar_distance(lidar_distances[i-1], lidar_distances[i], lidar_degs[i-1], lidar_distances[i]) > gap_threshold) or (i==len(lidar_distances)-1):
                end = i
                object_deg_ranges.append(begin, end)
                begin = end

        for begin, end in object_deg_ranges:
            points = lidar_distances[begin:end]
            degs = lidar_degs[begin:end]

            object_dist = np.average(points)
            object_angle = np.max(degs) - np.min(degs)

            object_coord = self.__polar_to_cartesian(object_dist, np.radians(object_angle))
            object_coords.append(object_coord)

        return object_coords


    def __polar_to_cartesian(self, r, theta):
        return r * np.cos(theta), r * np.sin(theta)

    def __check_if_on_track(self, object_coords):
        to_return = []
        for object in object_coords:
            pt = geom.Point(*object)
            if self.left_poly.contains(pt) and not self.right_poly.contains(pt):
                cte = pt.distance(self.cte_ring)
                if not self.cte_poly.contains(pt):
                    cte *= -1.0
                to_return.append(ObjectPack(object[0], 0, object[1], cte, True))
        return to_return



    def getName(self):
        return 'PointCloud Clustering (DonkeyGym)'
'''
class DonkeyGymFindGap(Component):
    def __init__(self, cfg):
        super().__init__(inputs=['lidar/valid_degs', 'lidar/valid_distances', 'gym/telemetry'], outputs=['detection/objects_on_track'], threaded=False)
        seg_data_path = cfg['seg_data_file']
        cte_data_path = cfg['cte_data_file']
        left_data_path = cfg['left_data_file']
        right_data_path= cfg['right_data_file']
        with open(seg_data_path, 'r') as input_file:
            self.seg_data = json.load(input_file)
        with open(cte_data_path, 'r') as input_file:
            self.cte_data = json.load(input_file)
        with open(left_data_path, 'r') as input_file:
            self.left_data = json.load(input_file)
        with open(right_data_path, 'r') as input_file:
            self.right_data = json.load(input_file)

        seg_coords = [(pt['gym/x'], pt['gym/z']) for pt in self.seg_data]
        cte_coords = [(pt['gym/x'], pt['gym/z']) for pt in self.cte_data]
        left_coords = [(pt['gym/x'], pt['gym/z']) for pt in self.left_data]
        right_coords = [(pt['gym/x'], pt['gym/z']) for pt in self.right_data]
        self.cte_headings = [pt['yaw'] for pt in self.cte_data]
        self.seg_path = geom.LineString(seg_coords)
        self.cte_ring = geom.LinearRing(cte_coords)
        self.cte_poly = geom.Polygon(cte_coords)
        self.left_poly = geom.Polygon(left_coords)
        self.right_poly = geom.Polygon(right_coords)

        

    def step(self, *args):
        lidar_degs, lidar_dists, tele = args
        if None in args:
            return [],
        lidar_degs = np.array(lidar_degs)
        lidar_dists = np.array(lidar_dists)
        idx = np.where((lidar_degs < 60) | (lidar_degs > 300))[0]
        lidar_degs = lidar_degs[idx]
        lidar_dists = lidar_dists[idx]

        heading = tele.gyro_y
        offset = np.array([tele.pos_x, tele.pos_z])
        coordinates = self.__polar_to_cartesian(lidar_dists, np.radians(- lidar_degs + heading))
        
        return self.__check_if_on_track(coordinates + offset[:,np.newaxis])

    def __polar_distance(self, r1, r2, theta1, theta2):
        return np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(theta2 - theta1))



    def __polar_to_cartesian(self, r, theta):
        result = np.vstack([r * np.cos(theta), r * np.sin(theta)])
        if result.shape == (2, 1): return result.T
        return result

    def __check_if_on_track(self, object_coords):
        to_return = []
        if object_coords is not None and object_coords.size > 1:
            for object in object_coords:
                pt = geom.Point(object[0], object[1])
                if self.left_poly.contains(pt) and not self.right_poly.contains(pt):
                    cte = pt.distance(self.cte_ring)
                    if not self.cte_poly.contains(pt):
                        cte *= -1.0
                    to_return.append(ObjectPack(object[0], 0, object[1], cte, True))
            #if len(to_return) > 0:
            #   for object in to_return:
            #        print(object.__dict__)
        return to_return,



    def getName(self):
        return 'DonkeyGymFindGap'