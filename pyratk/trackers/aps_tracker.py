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
    def __init__(self, daq, receiver_array):
        """
        Initialize tracker class.
        """
        # copy arguments into attributes
        self.daq = daq
        self.receiver_array = receiver_array
        self.detections = []

        self.pulse = self.receiver_array[0].transmitter.pulses[0]
        self.chirp_rate = self.pulse.bw / self.pulse.delay

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

        fft_mats = [self.receiver_array[0].fft_mat, self.receiver_array[1].fft_mat]
        var0=np.power(np.mean(self.receiver_array[0].fft_mat,axis=0),2)
        var1=np.power(np.mean(self.receiver_array[1].fft_mat,axis=0),2)
        #var0=signal.resample_poly(var00,4,1)
        #var1=signal.resample_poly(var01,4,1)
        f=np.linspace(-50000,50000,num=var0.size-1)
        r_d0=np.abs(f[np.argmax(var0)]*3e8/self.chirp_rate/2)
        r_d1=np.abs(f[np.argmax(var1)]*3e8/self.chirp_rate/2)
        theta=np.arcsin((r_d0-r_d1)/0.3864)+0.5*np.pi
        R=0.5*(r_d0+r_d1)

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
