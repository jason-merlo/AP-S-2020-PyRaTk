# -*- coding: utf-8 -*-
"""
Tracker Class.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import numpy as np                            # Storing data
from pyratk.datatypes.ts_data import TimeSeries                # storing data
from pyratk.datatypes.geometry import Point, Circle, Triangle  # storing geometric information
import itertools                              # 'triangulating' radar radii
import math
import sys

# DEBUG
USE_LPF = True


class Tracker2D(object):
    """Class to track detections using 4 doppler measurements."""

    def __init__(self, data_mgr, radar_array, dim=1, constraint='auto',
                 start_loc=Point(0.0, 0.0, 0.26035)):  # Height of top in m
        """
        Initialize tracker class.

        Args:
        data_mgr
            DataManager object
        radar_array
            Array of radar objects, tuple matching physical layout
        dim (default=1)
            Dimensionality of tracker
        constraint (defalt='auto')
            Spatial constraint for reduced dimensionality trackers
            'auto' will pick the first valid constraint in the list
        start_loc (default=Point(0.0, 0.0, 0.26035))
            Point indicating the initial location of the object to track
        """
        super(Tracker2D, self).__init__()

        self.valid_constraints = {1: ['x', 'y', 'z'],
                                  2: ['xy', 'xz', 'yz'],
                                  3: []}

        # Check constraints
        try:
            if constraint == 'auto':
                self.const = self.valid_constraints[dim][0]
            elif constraint in self.valid_constraints[dim]:
                self.const = constraint
            else:
                raise ValueError("Invalid constraint provided")
        except KeyError:
            raise ValueError("Invalid dimensionality provided")

        # copy arguments into attributes
        self.data_mgr = data_mgr
        self.array = radar_array
        self.start_loc = start_loc
        self.loc = self.start_loc.copy()

        init_length = 4096
        self.ts_track = TimeSeries(init_length, dtype=Point)

    def rho_to_r(self, rho, phi):
        """
        Compute 2D radius based on rho, phi, and z.

           r
        ______
        |    /
        |   /
        |  / rho
        |_/
        |/phi

        """
        return rho * np.sin(phi)

    def update_relative_positions(self, radar):
        """
        Update rho, phi, theta, and r, of the track relative to the radar.

        NOTE: Math varified
        """
        radar.rho_vec = self.loc - radar.loc
        radar.r.append(Point(radar.rho_vec.x, radar.rho_vec.y).length)

        radar.theta = np.arctan2(radar.rho_vec.y, radar.rho_vec.x)
        # print(radar.rho_vec)
        assert(radar.rho_vec.z > 0), 'Implausibility: rho_vec.z <= 0'
        radar.phi = np.arctan(radar.r / radar.rho_vec.z)

        # print('rho:{:+7.3}, phi:{:+7.3}, theta:{:+7.3}, r:{:+7.3}'.format(
        # radar.rho_vec.length, radar.phi, radar.theta, radar.r))

    def propagate_track_radius(self, radar):
        """
        Calculate new detection radius from current location and measurement.

        Based on the current location of the tracked object, the new radius
        is calculated by propagating the location by the integrated velocity
        measured by the radar over the sampling period.

        Args:
            radar
                Radar object to update
        """
        # Append new data
        if len(radar.ts_v) > 0 and USE_LPF:
            # Low-pass filter
            radial_vel = self.rho_to_r(
                radar.vmax, radar.phi) * 0.5 + radar.ts_v[-1] * 0.5
        else:
            radial_vel = self.rho_to_r(radar.vmax, radar.phi)
        radar.ts_v.append(radial_vel)

        # Need a time delta (two samples) before position can be updated
        if len(radar.ts_a) > 0:
            dt = 1.0 / radar.update_rate * 4  # Why is this multiplied by 4?
            r = radar.r  # radar.ts_r.data[-1]
            v = radar.ts_v.data[-1]
            a = radar.ts_a.data[-1]

            # Calculate radius and acceleration
            ap = (v - radar.ts_v.data[-2]) / dt
            rp = r + v * dt + 0.5 * a * dt**2
            radar.ts_a.append(ap)
            radar.ts_r.append(rp)
            # print('new values: \
            #       r: {: +7.3f}, v: {: +7.3f}, a: {: +7.3f}, dt: {: +7.3f}'
            #       .format(rp, v, ap, dt))
        else:
            # Append initial values
            radar.ts_r.append(radar.r)
            radar.ts_a.append(0)

    def update(self):
        """Update position of track based on differential updates."""
        # Loop through all new data that has arrived in the buffer
        buffer = self.data_mgr.buffer

        if not buffer:
            print("(DEBUG, tracker.py) Nothing in buffer.")
        else:
            print("(DEBUG, tracker.py) Updating tracker...")

        while buffer:
            # Remove oldest data in queue
            data, sample_index = buffer.pop(0)

            # Check for a virtual DAQ loop around
            if sample_index == 0:
                self.reset()

            # Compute new measurements at each radar based on new data
            self.array.update(data)

            # Check to see if this is the first iteration
            # Cannot update without two measurements
            if len(self.array.radars[0].ts_r) > 0:
                intersections = []
                # flatten radars list for combinations
                flat_array = []
                for ry in self.array.radars:
                    for rx in ry:
                        flat_array.append(rx)
                flat_array = flat_array[1:]

                # find intersections between radar circles
                for radar_pair in itertools.combinations(flat_array, 2):
                    print("tracker.py: radar_pair:", radar_pair)
                    # Get most recent radius data
                    r1 = radar_pair[0].ts_r.data[-1]
                    r2 = radar_pair[1].ts_r.data[-1]

                    # Get radar locations
                    p1 = radar_pair[0].loc
                    p2 = radar_pair[1].loc

                    # Create circle objects from radar information
                    c1 = Circle(p1, r1)
                    c2 = Circle(p2, r2)
                    # print(c1)
                    # print(c2)

                    # Calculate all intersections, or nearest approximation
                    intersect = c1.intersections(c2)  # TODO check for bias
                    # print(intersect)
                    # print('='*50)
                    intersections.append(intersect)

                # Find triangle with lowest area
                potentials = itertools.product(*intersections)
                lowest_area = -1
                best_triangle = Triangle()
                for p in potentials:
                    t = Triangle(*p)
                    # area = t.area  # TODO Cehck for zero bias
                    area = t.circumference
                    if (area < lowest_area or lowest_area == -1):
                        lowest_area = area
                        best_triangle = t

                # Set centroid of best triangle to new location
                self.best_triangle = best_triangle
                self.loc.x = best_triangle.centroid.x  # TODO check for bias
                # self.loc.y = best_triangle.centroid.y

                if not math.isnan(self.loc.x):
                    # print('Current location:', self.loc)
                    pass
                else:
                    print('Nan encountered, exiting...')
                    self.array.data_mgr.close()
                    sys.exit(0)

            # Append new location to track
            p = Point(*self.loc.p)
            self.ts_track.append(p)

            for i, radar in enumerate(self.array):
                # print("=== RADAR {:} ===".format(i))
                self.update_relative_positions(radar)  # Updates radar.r
                self.propagate_track_radius(radar)

    def reset(self):
        """Reset all temporal elements."""
        self.ts_track.clear()
        self.array.reset()
        self.loc = self.start_loc.copy()


# class TrackerEvaluator(Object):
#     def __init__():
