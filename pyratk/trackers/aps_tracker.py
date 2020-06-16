# -*- coding: utf-8 -*-
"""
AP-S Tracker Class.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import numpy as np                            # Storing data

from pyratk.datatypes.ts_data import TimeSeries                # storing data
from pyratk.datatypes.motion import StateMatrix
from pyratk.datatypes.geometry import Point
from pyratk.datatypes.radar import Detection

class ApsTracker(object):
    """Class to track detections using 4 doppler measurements."""

    # === INITIALIZATION METHODS ============================================= #
    def __init__(self, daq, receiver_array):
        """
        Initialize tracker class.
        """
        self.valid_constraints = {1: ['x', 'y', 'z'],
                                  2: ['xy', 'xz', 'yz'],
                                  3: []}

        # copy arguments into attributes
        self.daq = daq
        self.receiver_array = receiver_array
        self.detections = []

        # Configure control signals
        self.connect_control_signals()

    def connect_control_signals(self):
        """Initialize control signals."""
        self.receiver_array.data_available_signal.connect(self.update)
        self.daq.reset_signal.connect(self.reset)


    # ====== CONTROL METHODS ================================================= #
    def update(self):
        """
        Update position of track based on new data.

        Called by data_available_signal signal in DAQ.
        """

        self.detections.clear()

        # Add new Detection objects to detections list

        # loc is cylindrical (R, theta, Z), but Z is ignored by plot
        R = np.random.rand() * 15
        theta = np.random.rand() * np.pi
        loc = Point(R, theta, 0.0)
        new_detection = Detection(loc)
        self.detections.append(new_detection)

    def reset(self):
        """Reset all temporal elements."""
        print("(tracker.py) Resetting tracker...")
        self.detections.clear()

# class TrackerEvaluator(Object):
#     def __init__():
