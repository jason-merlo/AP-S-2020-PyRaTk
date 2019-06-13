# -*- coding: utf-8 -*-
"""
Radar Class.

Contains Radar class and a RadarTypes class

Contains RadarArray class to hold timeseries data for voltage measurements,
FFT slices, and max frequency.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import numpy as np              # Storing data
from ts_data import TimeSeries  # storing data
import scipy.constants as spc   # speed of light
from pyratk.datatypes.geometry import Point      # radar location
import itertools                # flatten radar array for indexing
from collections import deque   # Used for keeping previous states


# CONSTANTS
POWER_THRESHOLD = 4  # dBm


class Radar(object):
    """
    Object to hold Radar information and signal processing.

    It is assumed radars will have both I and Q channels.

    Attributes:
        data_mgr
            data_mgr_mgr object
        ts_data
        ts_drho
        cfft_data
        loc
            Point containing relative location

    """

    def __init__(self, data_mgr, data_idx, loc=Point(),
                 f0=24.150e9, fft_size=2**16, fft_win_size=2**12):
        super(Radar, self).__init__()
        """
        Initialize radar object.

        Args:
            data_mgr
                [data_mgr] object for DAQ configurations
            data_idx
                Index of radar channels in DAQ output.  The I and Q channels
                must be adjacent (ex. Ch. 1 --> 1 & 2, Ch. 2 --> 3 & 4...)
            loc
                Point() indicating the location of the radar
        """

        # Data stream parameters
        # TODO: create DataStream object representing I or I/Q data
        self.data_mgr = data_mgr
        self.index = data_idx

        # Physical parameters
        self.f0 = f0
        self.loc = loc

        # Processing parameters
        self.fft_size = fft_size

        # Derived processing parameters
        self.update_rate = self.data_mgr.sample_rate / self.data_mgr.sample_size
        self.bin_size = self.data_mgr.sample_rate / self.fft_size
        self.center_bin = np.ceil(self.fft_size / 2)

        # Set fft windowing size for zero-padding (default is 4x fft_size)
        if fft_win_size is None:
            self.window_size = 4 * fft_size
        else:
            self.window_size = fft_win_size

        # === State variables ===
        # initial array size 4096 samples
        length = 4096
        data_shape = (2, self.data_mgr.source.sample_size)

        # initialize data arrays
        # NOTE: cfft_data initialize to ones for log graph – log(0) is undef.
        self.ts_data = TimeSeries(length, data_shape, dtype=np.complex64)
        self.cfft_data = np.ones(self.window_size, dtype=np.float64)
        self.fft_window = np.empty((self.window_size, 2), dtype=np.float64)

        # Instantaneous state variables
        self.fmax = 0

        # Initialize kinematics timeseries
        self.rho_dot = 0
        self.ts_drho = TimeSeries(length)

        # TODO Depricate
        self.ts_r = TimeSeries(length)
        self.ts_v = TimeSeries(length)
        self.ts_a = TimeSeries(length)

    def freq_to_vel(self, freq):
        """Compute the velocity for a given frequency and the radar f0."""
        c = spc.speed_of_light

        vel = (c * self.f0 / (freq + self.f0)) - c

        return vel

    def bin_to_freq(self, bin):
        """Compute frequency based on bin location."""
        return (bin - self.center_bin) * self.bin_size

    def compute_cfft(self, data, fft_size):
        """Compute fft and fft magnitude for plotting."""
        # Create complex data from input
        complex_data = data[0] + data[1] * 1j
        # Create hanning window
        hanning = np.hanning(complex_data.shape[0])
        fft_complex = np.fft.fft(complex_data * hanning, fft_size)
        # Display only magnitude
        fft_mag = np.linalg.norm([fft_complex.real, fft_complex.imag], axis=0)

        # Adjust fft so DC is at the center
        center = int(fft_mag.shape[0] / 2)
        fft_data = np.empty(fft_mag.shape)
        fft_data[:center] = fft_mag[center:]
        fft_data[center:] = fft_mag[:center]

        return fft_data

    def update(self, data):
        # Get data from data_mgr
        slice = 2 * self.index

        # TODO remove ts_data, use data_mgr.ts_buffer instead
        self.ts_data.append(data[slice:slice + 2])

        # Get window of FFT data
        # TODO: why didn't the below line work?
        # window_slice = \
        #     self.data_mgr.ts_buffer[slice:slice + 2][-self.window_size // self.data_mgr.sample_size:]
        window_slice = \
                self.ts_data[-self.window_size // self.data_mgr.sample_size:]
        slice_shape = window_slice.shape
        start_idx = (slice_shape[0] * slice_shape[2]) - self.window_size
        # Check if time-series is still smaller than window size
        if start_idx < 0:
            start_idx = 0

        i_data = []
        q_data = []
        for i in range(slice_shape[0]):
            i_data.append(window_slice[i][0])
            q_data.append(window_slice[i][1])

        i_data = list(itertools.chain(*i_data))
        q_data = list(itertools.chain(*q_data))
        iq_data = np.array([i_data, q_data])

        # Calculate complex FFT (may be zero-padded if fft-size > sample_size)
        self.cfft_data = self.compute_cfft(iq_data, self.fft_size)

        # Power Thresholding
        vmax_bin = np.argmax(self.cfft_data).astype(np.int32)
        if self.cfft_data[vmax_bin] < POWER_THRESHOLD:
            self.fmax = 0
        else:
            self.fmax = self.bin_to_freq(vmax_bin)

        # Add current measurement to time series
        self.ts_drho.append(self.vmax)
        self.drho = self.vmax


    def reset(self):
        pass
        # self.ts_data.clear()
        # self.ts_drho.clear()
        # self.ts_v.clear()
        # self.ts_r.clear()
        # self.ts_a.clear()

    @property
    def vmax(self):
        return -self.freq_to_vel(self.fmax)


class RadarArray(list):
    """
    Hold timeseries data of array measurements.

    Attributes:
        radars
            A list of radar objects in the array

    """

    def __init__(self, radar_list):
        """
        Initialize radar array.

        Args:
            radar_list
                List of Radar objects in array
        """
        # copy arguments into member variables
        self.radars = radar_list
        self.initial_update = True

        # Used for iterator magic functions
        self.i = 0

    def reset(self):
        """Clear all temporal data from radars in array."""
        for radar in self.radars:
            radar.reset()

    def update(self, data):
        """Update all radars in array."""
        for radar in self.radars:
            radar.update(data)
