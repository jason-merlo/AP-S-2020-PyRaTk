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
                 start_loc=StateMatrix(mat=np.array([[0.0, 0.0, 0.0],
                                                     [0.0, 0.0, 0.0],
                                                     [0.26035, 0.0, 0.0]]),
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
        self.state = StateMatrix(self.start_loc.copy())

        init_track_length = 4096
        self.ts_location = TimeSeries(init_track_length,
                                      (3, 3),
                                      dtype=np.float64)

        # append initial location
        self.ts_location.append(self.state.q)

        # Configure control signals
        self.connect_control_signals()

    def connect_control_signals(self):
        """Initialize control signals."""
        self.radar_array.data_available_signal.connect(self.update)
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
    def update_fused_state_estimate(self):
        """
        Compute the fused of state estimate from all radars in array.

        Currently fusion method is averaging all motion vectors.
        Assumptions:
            z-height is constant.
        """
        average_velocity_vector = Point()

        # Compute intersecting perpendicular lines - Y = Ax + B
        Y = []
        A = []
        B = []

        for radar in self.radar_array:
            # Compute unit vector from radar towards the target
            rho = Point(*self.state.q[:, 0]) - radar.loc
            r = Point(rho[0], rho[1])  # Project onto x-y plane
            r_hat = r.copy()
            r_hat.normalize()

            # Compute elevation angle relative to radar
            # # Compute angle between track and z-axis for each radar
            phi = np.arctan2(r.length, rho.z)

            # Compute angle between target bearing and radar
            # psi = np.arccos(r.length, )

            # Assign vector length to measured Doppler velocity
            dr = r_hat * radar.drho / np.sin(phi)
            # print("dr:\t", dr)

            # Add radar measurement vector to average
            average_velocity_vector += dr

        average_velocity_vector /= len(self.radar_array)

        # Update state matrix based on fused data
        self.state.q[:, 1] = average_velocity_vector
        self.state.q[:, 0] += self.state.q[:, 1] * self.data_mgr.source.update_period

        # print('(tracker.py) v_bar:\t', self.state.q[:, 1])
        # print(self.state)

    # ====== CONTROL METHODS ================================================= #
    def update(self):
        """
        Update position of track based on differential updates.

        Called by data_available_signal signal in DAQ.
        """

        # Compute new track location
        self.update_fused_state_estimate()

        # Add state matrix to TimeSeries
        self.ts_location.append(self.state.q)

        # print("(tracker.py) initial_location: \n", self.ts_location[0])

    def reset(self):
        """Reset all temporal elements."""
        print("(tracker.py) Resetting tracker...", self.start_loc.copy())
        self.state.q = self.start_loc.copy()
        self.ts_location.clear()
        self.ts_location.append(self.state.q)

# class TrackerEvaluator(Object):
#     def __init__():
