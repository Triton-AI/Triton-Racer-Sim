""" 
An alternative approach to neural network AI pilot
using pid control to follow a predefined race line.

Usage:
1. Drive around the track, carefully following the optimal race line;
2. Process into a track data;
3. Turn on location tracker in myconfig;
4. Set model_type to "pid" in myconfig;
5. Drive.
"""
from operator import ne
from simple_pid import PID
import numpy as np
import json

from TritonRacerSim.components.component import Component
from TritonRacerSim.components.controller import DriveMode
from sklearn.neighbors import NearestNeighbors


class PIDPilot(Component):
    def __init__(self, cfg):
        Component.__init__(self, inputs=['gym/speed', 'loc/cte', 'usr/mode', 'loc/break_indicator', 'loc/heading',
                                         'gym/telemetry', 'detection/objects_on_track'], outputs=['ai/steering', 'ai/throttle', 'ai/pid'], threaded=False)
        str_cfg = cfg['steering']
        spd_cfg = cfg['speed']

        # Input CTE to optimal racing line; Output steering
        self.str_pid = PID(str_cfg['kp'], str_cfg['ki'], str_cfg['kd'],
                           setpoint=0.0, output_limits=(-1.0, 1.0), sample_time=0.05)

        # Input difference between the current speed and desired speed; Output throttle
        self.spd_pid = PID(spd_cfg['kp'], spd_cfg['ki'], spd_cfg['kd'],
                           setpoint=0.0, output_limits=(0.0, 1.0), sample_time=0.05)

        self.k = cfg['speed_limit_k']
        self.prev_mode = DriveMode.HUMAN

    def step(self, *args):
        spd, cte, mode, ind, heading_pred, tele, objects = args
        if None not in args[0:6]:
            if self.prev_mode != DriveMode.AI and mode == DriveMode.AI:
                self.str_pid.reset()
                self.spd_pid.reset()
            self.prev_mode = mode
            heading_diff = self.__calc_heading_difference(
                tele.yaw, heading_pred)
            heading_diff_normalized = heading_diff / (2 * np.pi) / 3
            cte = self.__correct_cte(cte, objects)
            cte_normalized = cte * 2 / 3

            str = self.str_pid(cte_normalized + heading_diff_normalized)

            d_spd = spd - (self.k / abs(str) + 5.0) if str != 0.0 else -1
            thr = self.spd_pid(d_spd)
            cte_str = "{0:0.4f}".format(cte)
            # print (f'CTE: {cte}, Str: {str}, Thr: {thr}\r', end='')
            if ind == 1:
                thr = 0.0
            return str, thr, self
        else:
            return 0.0, 0.0, self

    def __calc_heading_difference(self, curr_h, pred_h):
        # negative for turning left, positive for tunring right
        if pred_h > curr_h:
            tl = pred_h - curr_h
            tr = curr_h + 360 - pred_h
            return -tl if tl < tr else tr
        else:
            tl = 360 - curr_h + pred_h
            tr = curr_h - pred_h
            return -tl if tl < tr else tr

    def __correct_cte(self, cte, objects):
        clearance = 1
        return cte

        if objects is None or len(objects) == 0:
            return cte

        obj = objects[0]
        if abs(obj.cte) > clearance:
            return cte

        margin = abs(abs(obj.cte) - clearance)

        if (cte <= 0 and obj.cte >= 0) or (cte >= 0 and obj.cte <= 0):
            return cte - margin
        else:
            return cte + margin

    def getName(self):
        return 'PID Pilot'


class WaypointFollower(Component):
    """
    PID Pilot Mark II. This class does not require the track to be non-self-intersecting.
    """

    def __init__(self, cfg_pid, cfg_waypoint):
        Component.__init__(self, inputs=['gym/telemetry', 'usr/mode'], outputs=[
                           'ai/steering', 'ai/throttle', 'ai/pid'], threaded=False)
        str_cfg = cfg_pid['steering']
        spd_cfg = cfg_pid['speed']

        # Input CTE to optimal racing line; Output steering
        self.str_pid = PID(str_cfg['kp'], str_cfg['ki'], str_cfg['kd'],
                           setpoint=0.0, output_limits=(-1.0, 1.0), sample_time=0.05)
        # Input difference between the current speed and desired speed; Output throttle
        self.spd_pid = PID(spd_cfg['kp'], spd_cfg['ki'], spd_cfg['kd'],
                           setpoint=0.0, output_limits=(-0.1, 1.0), sample_time=0.05)

        self.prev_mode = DriveMode.HUMAN

        wp_data_path = cfg_waypoint['waypoint_data_file']
        with open(wp_data_path, 'r') as input_file:
            self.wps = json.load(input_file)

        self.wp_coords = np.array(
            [(pt['gym/z'], pt['gym/x']) for pt in self.wps])
        self.wp_headings = np.array([pt['yaw'] for pt in self.wps])
        self.wp_speeds = np.array([pt['speed'] for pt in self.wps])

        self.last_nearest_wp_idx = None
        self.n_look_ahead = cfg_waypoint['n_look_ahead']

    def step(self, *args):
        tele, mode = args

        if self.prev_mode != DriveMode.AI and mode == DriveMode.AI:
            self.str_pid.reset()
            self.spd_pid.reset()
            self.last_nearest_wp_idx = None
        self.prev_mode = mode

        if tele is not None:
            speed = tele.speed
            heading = tele.yaw
            coord = tele.pos_z, tele.pos_x

            target_heading, target_speed = self.__plan(coord)
            #print("{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:d}".format(
            #    target_heading, heading, coord[0], coord[1], self.last_nearest_wp_idx))
            return self.__act(heading, target_heading, speed, target_speed)
        return 0.0, 0.0, self

    def __findNearestWaypoint(self, coord, waypoints=None):
        """
        coord: (x, y)
        waypoints: np.array (N * 2)

        Exhaustively find the nearest waypoint.
        Return: the index of the nearest neighbor in self.wps
        """
        if waypoints is None:
            waypoints = self.wp_coords
        coord_arr = np.expand_dims(np.array(coord), axis=0)
        dists = coord_arr - waypoints
        return np.argmin(np.linalg.norm(dists, axis=1))

    def __lookAhead(self, idx, n):
        """
        Return the indices of the n waypoints ahead (including the current one), 
        considering the end and begining of a lap 
        """
        num_wps = len(self.wps)
        if num_wps - idx >= n:
            return np.linspace(idx, idx+n, n, endpoint=False, dtype=int)
        else:
            return np.hstack(
                (np.linspace(idx, num_wps, num_wps-idx, endpoint=False, dtype=int),
                 np.linspace(0, n-(num_wps-idx), n-(num_wps-idx),
                             endpoint=False, dtype=int)
                 )
            )

    def __lookBack(self, idx, n):
        """
        Return the indices of the n waypoints back (including the current one), 
        considering the end and begining of a lap 
        """
        num_wps = len(self.wps)
        if n <= idx+1:
            return np.linspace(idx, idx-n, n, endpoint=False, dtype=int)
        else:
            return np.hstack(
                (np.linspace(idx, 0, idx+1, endpoint=True, dtype=int),
                 np.linspace(num_wps-1, num_wps-(n-idx), n -
                             idx-1, endpoint=False, dtype=int)
                 )
            )

    def __efficientFindNearestWaypoint(self, coord, last_waypoint_idx):
        """
        coord: (x, y)
        last_waypoint_idx: int

        Efficiently find the nearest waypoint when possible, given the last nearest waypoint
        Return: the index of the nearest neighbor in self.wps
        """
        roi = 30
        nearby_wp_indices = self.__getNeighbors(last_waypoint_idx, roi)
        nearest_wp_idx = nearby_wp_indices[self.__findNearestWaypoint(
            coord, self.wp_coords[nearby_wp_indices])]

        # If the nearest neighbor is at the boundary of the ROI, we need to exhaustively search.
        if nearest_wp_idx == nearby_wp_indices[0] or nearest_wp_idx == nearby_wp_indices[-1]:
            return self.__findNearestWaypoint(coord)
        else:
            return nearest_wp_idx

    def __getNeighbors(self, idx, roi):
        """ Return roi * index of neighbor waypoints"""
        result = np.hstack(
            (np.flip(self.__lookBack(idx, roi)), self.__lookAhead(idx, roi)))
        return np.hstack((result[:roi-1], result[roi:]))

    def __getTargetHeading(self, target_wp_idx, coord):
        """
        Return the target heading in degree
        """
        target_wp_coord = self.wp_coords[target_wp_idx]
        target_heading = np.degrees(np.arctan2(
            target_wp_coord[1]-coord[1], target_wp_coord[0]-coord[0]))
        if target_heading < 0:
            target_heading = 360 + target_heading
        return target_heading

    def __plan(self, coord):

        #self.last_nearest_wp_idx = self.__findNearestWaypoint(coord)
        # OR

        if self.last_nearest_wp_idx is None:
            self.last_nearest_wp_idx = self.__findNearestWaypoint(coord)
        else:
            self.last_nearest_wp_idx = self.__efficientFindNearestWaypoint(
                coord, self.last_nearest_wp_idx)

        target_wp_idx = self.__lookAhead(
            self.last_nearest_wp_idx, self.n_look_ahead)[-1]
        target_heading = self.__getTargetHeading(
            target_wp_idx, coord)
        target_speed = self.wp_speeds[target_wp_idx]

        return target_heading, target_speed

    def __act(self, current_heading, target_heading, current_speed, target_speed):

        heading_diff = self.__calc_heading_difference(
            current_heading, target_heading)  # in [-180, 180]
        # print(heading_diff)
        heading_diff_normalized = heading_diff / 180  # normalize to [-1, 1]

        str = self.str_pid(heading_diff_normalized)

        d_spd = current_speed - target_speed
        thr = self.spd_pid(d_spd)

        return str, thr, self

    def __calc_heading_difference(self, curr_h, pred_h):
        # negative for turning left, positive for tunring right
        if pred_h > curr_h:
            tl = pred_h - curr_h
            tr = curr_h + 360 - pred_h
            return -tl if tl < tr else tr
        else:
            tl = 360 - curr_h + pred_h
            tr = curr_h - pred_h
            return -tl if tl < tr else tr

    def getName(self):
        return 'Waypoint Follower'
