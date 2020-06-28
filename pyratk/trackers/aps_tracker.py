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
from scipy import signal

class ApsTracker(object):
    """Class to track detections using 4 doppler measurements."""

    # === INITIALIZATION METHODS ============================================= #
    def __init__(self, daq, receiver_array, moving_average_weight=1.0):
        """
        Initialize tracker class.
        """
        # copy arguments into attributes
        self.daq = daq
        self.receiver_array = receiver_array
        self.detections = []

        self.max_freq = np.zeros(len(receiver_array))
        self.max_range = np.zeros(len(receiver_array))

        self.weight = moving_average_weight

        self.pulse = self.receiver_array[0].transmitter.pulses[0]
        self.chirp_rate = self.pulse.bw / self.pulse.delay

        self.baseline=1.3 # m

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
        self.max_range *= 1.0 - self.weight

        # Add new Detection objects to detections list

        fft_mats = [self.receiver_array[0].fft_mat, self.receiver_array[1].fft_mat]
        var0 = np.power(np.mean(self.receiver_array[0].fft_mat,axis=0),2)
        var1 = np.power(np.mean(self.receiver_array[1].fft_mat,axis=0),2)
        #var0=signal.resample_poly(var00,4,1)
        #var1=signal.resample_poly(var01,4,1)
        self.max_freq[0] = (np.argmax(var0, axis=0) - self.receiver_array[0].fast_center_bin) * self.receiver_array[0].fast_bin_size
        self.max_freq[1] = (np.argmax(var1, axis=0) - self.receiver_array[1].fast_center_bin) * self.receiver_array[1].fast_bin_size

        self.max_range[0] += (np.abs(self.max_freq[0] * 3e8/self.chirp_rate/2) - 2.47) * self.weight
        self.max_range[1] += (np.abs(self.max_freq[1] * 3e8/self.chirp_rate/2) - 2.47) * self.weight

        theta = np.arcsin((self.max_range[0] - self.max_range[1]) / self.baseline) + np.pi * 0.5
        R = np.average(self.max_range) * 0.5

        # loc is cylindrical (R, theta, Z), but Z is ignored by plot
        #R = np.random.rand() * 15
        #theta = np.random.rand() * np.pi
        loc = Point(R, theta, 0.0)
        new_detection = Detection(loc)
        self.detections.append(new_detection)

    def reset(self):
        """Reset all temporal elements."""
        print("(tracker.py) Resetting tracker...")
        self.detections.clear()

# class TrackerEvaluator(Object):
#     def __init__():
