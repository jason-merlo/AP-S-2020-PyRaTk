# -*- coding: utf-8 -*-
"""
Tracker Class.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import numpy as np                            # Storing data
from pyratk.datatypes.ts_data import TimeSeries                # storing data
from pyratk.datatypes.motion import StateMatrix
from pyratk.datatypes.geometry import Point

# DEBUG
USE_LPF = True


class Tracker2D(object):
    """Class to track detections using 4 doppler measurements."""

    # === INITIALIZATION METHODS ============================================= #
    def __init__(self, data_mgr, radar_array, dim=1, constraint='auto',
                 start_loc=StateMatrix(mat=np.array([[0.0, 0.0, 0.26035],
                                                     [0.0, 0.0,     0.0],
                                                     [0.0, 0.0,     0.0]]),
                                       coordinate_type='cartesian')):
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
        self.radar_array = radar_array
        self.start_loc = start_loc.q.copy()
        self.location = StateMatrix(self.start_loc.copy())

        init_track_length = 4096
        self.ts_location = TimeSeries(init_track_length,
                                      (3, 3),
                                      dtype=np.float64)

        # Configure control signals
        self.connect_control_signals()

    def connect_control_signals(self):
        """Initialize control signals."""
        self.data_mgr.data_available_signal.connect(self.update)
        self.data_mgr.reset_signal.connect(self.reset)

    # === HELPER METHODS ===================================================== #
    def rho_to_r(self, rho, phi):
        """
        Compute 2D radius based on rho and phi.

           r
        ______
        |    /
        |   /
        |  / rho
        |_/
        |/phi

        """
        return rho * np.sin(phi)

    # ====== TRACKING METHODS ================================================ #
    def update_fused_state_estimate(self, data):
        """
        Compute the fused of state estimate from all radars in array.

        Currently fusion method is averaging all motion vectors.
        Assumptions:
            z-height is constant.
        """
        average_velocity_vector = Point()

        for radar in self.radar_array:
            # Compute unit vector from radar towards track location
            rho = Point(*self.location[:, 0]) - radar.loc
            r_hat = Point(rho[0], rho[1])  # Constrain z
            r_hat.normalize()

            # print('(tracker.py) radar', radar.index, 'r_hat', r_hat)

            # Compute angle between track and z-axis for each radar
            phi = np.arctan2(np.sqrt(rho.x**2 + rho.y**2), rho.z)

            # Assign vector length to measured Doppler velocity
            dr = r_hat * radar.drho / np.cos(np.pi - phi)

            # Add radar measurement vector to average
            average_velocity_vector += dr

        average_velocity_vector /= len(self.radar_array)


        # print('(tracker.py) v_bar:\n', average_velocity_vector)

        # Update state matrix based on fused data
        self.location.q[:, 1] = average_velocity_vector * 4
        self.location.q[:, 0] += self.location.q[:, 1] * self.data_mgr.source.update_period
        # print(self.location)

    # ====== CONTROL METHODS ================================================= #
    def update(self, data_tuple):
        """
        Update position of track based on differential updates.

        Called by data_available_signal signal in DAQ.

        Args:
            data_tuple (tuple) - tuple holding (sample_array, sample_number)
        """
        # Extract new data from tuple and update location estimate
        data, sample_index = data_tuple

        # Compute new track location
        self.update_fused_state_estimate(data)

        # Add state matrix to TimeSeries
        self.ts_location.append(self.location.q)

        print("(tracker.py) initial_location: ",self.ts_location[0])

    def reset(self):
        """Reset all temporal elements."""
        print("(tracker.py) Resetting tracker...")
        self.location.q = self.start_loc.copy()
        self.ts_location.clear()

# class TrackerEvaluator(Object):
#     def __init__():
