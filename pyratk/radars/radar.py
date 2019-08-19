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
from pyratk.datatypes.ts_data import TimeSeries  # storing data
import scipy.constants as spc   # speed of light
from pyratk.datatypes.geometry import Point      # radar location
# import itertools                # flatten radar array for indexing
# from collections import deque   # Used for keeping previous states
# from reikna import fft          # Used for hardware acceleration of FFT
import time
# import logging
# import threading
from pyqtgraph import QtCore


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
                 f0=24.150e9, fft_size=2**16, fft_win_size=2**12,
                 cluda_thread=None):
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
        self.window_size = fft_win_size

        # Derived processing parameters
        self.update_rate = self.data_mgr.sample_rate / self.data_mgr.sample_chunk_size
        self.center_bin = np.ceil(self.fft_size / 2)
        self.bin_size = self.data_mgr.sample_rate / self.fft_size

        # === State variables ===
        # initial array size 4096 samples
        length = 4096
        chunk_size = self.data_mgr.source.sample_chunk_size
        data_shape = (2, chunk_size)

        # initialize data arrays
        # NOTE: cfft_data initialize to ones for log graph – log(0) is undef.
        self.ts_data = TimeSeries(length, data_shape, dtype=np.complex64)
        self.data_buffer = TimeSeries(length, (1,), dtype=np.complex64)
        self.cfft_data = np.ones(self.fft_size, dtype=np.float64)

        # Instantaneous state variables
        self.fmax = 0

        # Initialize kinematics timeseries
        self.rho_dot = 0
        self.ts_drho = TimeSeries(length)

        # TODO Depricate
        self.ts_r = TimeSeries(length)
        self.ts_v = TimeSeries(length)
        self.ts_a = TimeSeries(length)

        # Initialize hardare accelration
        self.cluda_thread = cluda_thread
        # if self.cluda_thread is not None:
        #     print('Configuring FFT...')
        #     reikna_fft = fft.FFT(np.empty(fft_size, dtype=np.complex64))
        #     print('Compiling FFT...')
        #     self.compiled_fft = reikna_fft.compile(cluda_thread)
        #     print('Configuring FFT Completed.')

    def freq_to_vel(self, freq):
        """Compute the velocity for a given frequency and the radar f0."""
        c = spc.speed_of_light
        velocity = (c * self.f0 / (freq + self.f0)) - c
        return velocity

    def bin_to_freq(self, bin):
        """Compute frequency based on bin location."""
        return (bin - self.center_bin) * float(self.bin_size)

    def compute_cfft(self, complex_data, fft_size):
        """Compute fft and fft magnitude for plotting."""
        # Create hanning window
        # hanning = np.hanning(complex_data.shape[0])

        # Create zero-padded array to be transformed
        fft_array = np.zeros((fft_size,), dtype=np.complex64)
        fft_array[:complex_data.size] = complex_data  # * hanning

        # Compute FFT using desired compute method
        if self.cluda_thread is None or True:
            # Normalize FFT magnitude to window size
            fft_complex = np.fft.fft(fft_array, norm='ortho')
        # else:
            # Currently not working
            # arr_dev = self.cluda_thread.to_device(fft_array)
            # res_dev = self.cluda_thread.array(fft_array.shape, fft_array.dtype)
            # self.compiled_fft(res_dev, arr_dev)
            # fft_complex = res_dev.get()

        # Adjust fft so DC is at the center
        fft_complex = np.fft.fftshift(fft_complex)

        # Display only magnitude
        fft_mag = np.linalg.norm([fft_complex.real, fft_complex.imag], axis=0)

        return fft_mag

    def update(self, data):
        # Get data from data_mgr
        channel_slice = 2 * self.index
        data_slice = data[channel_slice:channel_slice + 2]
        # iq_data_slice = data_slice[0, :] + data_slice[1, :] * 1.0j

        # TODO remove ts_data, use data_mgr.ts_buffer instead
        self.ts_data.append(data_slice)
        # self.data_buffer = np.append(self.data_buffer, iq_data_slice)

        # Get window of FFT data
        # window_slice = self.data_buffer[-self.window_size:]
        window_idx = self.window_size // self.data_mgr.sample_chunk_size
        window_slice_pair = self.ts_data[-window_idx:, ...]
        # print(window_slice_pair.shape)
        window_slice = window_slice_pair[:, 0, :].flatten() \
            + window_slice_pair[:, 1, :].flatten() * 1.0j
        # print(window_slice.shape)

        # Calculate complex FFT (may be zero-padded if fft-size > sample_chunk_size)
        # start_time = time.time()
        self.cfft_data = self.compute_cfft(window_slice, self.fft_size)
        # print('(radar.py) compute_cfft time: ', time.time() - start_time)

        # Find maximum frequency
        fmax_bin = np.argmax(self.cfft_data)
        self.fmax = self.bin_to_freq(fmax_bin)
        # Power Thresholding
        # if self.cfft_data[vmax_bin] < POWER_THRESHOLD:
        #     self.fmax = 0
        # else:
        #     self.fmax = self.bin_to_freq(vmax_bin)
        self.vmax = self.freq_to_vel(self.fmax)

        # Add current measurement to time series
        self.ts_drho.append(self.vmax)
        self.drho = self.vmax

    def reset(self):
        self.ts_data.clear()
        self.ts_drho.clear()


# class RadarArray(QtCore.QObject):
#     """
#     Hold timeseries data of array measurements.
#
#     Attributes:
#         radars
#             A list of radar objects in the array
#
#     """
#
#     def __init__(self, data_mgr, radar_list):
#         """
#         Initialize radar array.
#
#         Args:
#             radar_list
#                 List of Radar objects in array
#         """
#         super().__init__()
#
#         # copy arguments into member variables
#         self.data_mgr = data_mgr
#         self.radars = radar_list
#         # self.initial_update = True
#
#         # Used for iterator magic functions
#         self.idx = 0
#
#         # Flag to iqnore any stale data after a reset
#         self.reset_flag = False
#
#         # Configure Signals
#         self.connect_signals()
#
#     def connect_signals(self):
#         self.data_mgr.data_available_signal.connect(self.update)
#         self.data_mgr.reset_signal.connect(self.reset)
#
#     def reset(self):
#         """Clear all temporal data from radars in array."""
#         for radar in self.radars:
#             radar.reset()
#         self.reset_flag = True
#
#     def update(self, data_tuple):
#         """Update all radars in array."""
#         # start_time = time.time()
#         data, sample_index = data_tuple
#
#         print('(radar.py) sample_num:', sample_index)
#
#         if not self.reset_flag or sample_index == 0:
#             self.reset_flag = False
#             for radar in self.radars:
#                 radar.update(data)
#
#             # Emit data available signal for dependant tasks
#             self.data_available_signal.emit()
#         else:
#             print('(radar.py) ignoring stale data...')
#         # print('(radar.py) radar_array.update() ran in {:} (s)'
#         #       .format(time.time() - start_time))
#
#     # Iterable functions
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         try:
#             return_value = self.radars[self.idx]
#         except IndexError:
#             raise StopIteration
#             self.idx = 0
#         return return_value

class RadarArray(QtCore.QObject):
    """
    Hold timeseries data of array measurements.

    Attributes:
        radars
            A list of radar objects in the array

    """
    data_available_signal = QtCore.pyqtSignal()

    def __init__(self, data_mgr, radar_list):
        """
        Initialize radar array.

        Args:
            radar_list
                List of Radar objects in array
        """
        super().__init__()

        # copy arguments into member variables
        self.data_mgr = data_mgr
        self.radars = radar_list
        # self.initial_update = True

        # Used for iterator magic functions
        self.idx = 0

        # Flag to iqnore any stale data after a reset
        self.last_sample_index = -1

        # Configure Signals
        self.connect_signals()

    def connect_signals(self):
        """Connect signals for event driven operation."""
        self.data_mgr.data_available_signal.connect(self.update)
        self.data_mgr.reset_signal.connect(self.reset)

    def reset(self):
        """Clear all temporal data from radars in array."""
        for radar in self.radars:
            radar.reset()
        self.last_sample_index = -1

    def update(self, data_tuple):
        """Update all radars in array."""
        # start_time = time.time()
        data, sample_index = data_tuple

        print('(radar.py) sample_num:', sample_index)

        if sample_index == self.last_sample_index + 1:
            self.reset_flag = False
            for radar in self.radars:
                radar.update(data)

            # Emit data available signal for dependant tasks
            self.data_available_signal.emit()
        else:
            print('(radar.py) ignoring stale data...')

        self.last_sample_index = sample_index
        # print('(radar.py) radar_array.update() ran in {:} (s)'
        #       .format(time.time() - start_time))

    def __getitem__(self, i):
        """Return radar in array."""
        assert(i < len(self)), "index greater than flattened array size"
        return self.radars[i]

    def __len__(self):
        """Return length of radar array."""
        # compute sumproduct of array shape
        return len(self.radars)

    def __iter__(self):
        """Return self for iterator."""
        return self

    def __next__(self):
        """Return next radar object for iterator."""
        if self.idx < len(self):
            self.idx += 1
            return self[self.idx - 1]
        raise StopIteration()
