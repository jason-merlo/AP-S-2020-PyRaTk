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
from pyratk.datatypes.ts_data import TimeSeries     # storing data
import scipy.constants as spc   # speed of light
from pyratk.datatypes.geometry import Point         # radar location
# from pyratk.datatypes.motion import StateMatrix   # track location
from pyqtgraph import QtCore

from pyratk.formatting import warning


# CONSTANTS
POWER_THRESHOLD = 4  # dBm


class Radar(object):
    """
    Object to hold Radar information and signal processing.

    It is assumed radars will have both I and Q channels.

    NOTE: Radar class should not contain state history (TimeSeries).
          Work on removing these and placing them in Tracker or DataLogger.

    """

    # === INITIALIZATION METHODS ==============================================
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
        # negative --> towards radar; positive --> away from radar
        self.drho = 0

    #     self.connect_signals()
    #
    # def connect_signals(self):
    #     self.data_mgr.reset_signal.connect(self.reset)

    # === HELPER METHODS ======================================================
    def freq_to_vel(self, freq):
        """Compute the velocity for a given frequency and the radar f0."""
        c = spc.speed_of_light
        # velocity = (c * self.f0 / (freq + self.f0)) - c
        velocity = (freq * c) / (2 * self.f0)

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

        # Normalize FFT magnitude to window size
        fft_complex = np.fft.fft(fft_array, norm='ortho')

        # Adjust fft so DC is at the center
        fft_complex = np.fft.fftshift(fft_complex)

        # Display only magnitude
        fft_mag = np.linalg.norm([fft_complex.real, fft_complex.imag], axis=0)

        return fft_mag

    # === CONTROL METHODS =====================================================
    def update(self, data):
        # Get data from data_mgr
        channel_slice = 2 * self.index
        data_slice = data[channel_slice:channel_slice + 2]

        # TODO remove ts_data, use data_mgr.ts_buffer instead
        self.ts_data.append(data_slice)
        # self.data_buffer = np.append(self.data_buffer, iq_data_slice)

        # Get window of FFT data
        window_idx = self.window_size // self.data_mgr.sample_chunk_size
        window_slice_pair = self.ts_data[-window_idx:, ...]
        window_slice = window_slice_pair[:, 0, :].flatten() \
            + window_slice_pair[:, 1, :].flatten() * 1.0j

        # Calculate complex FFT
        # may be zero-padded if fft-size > sample_chunk_size
        self.cfft_data = self.compute_cfft(window_slice, self.fft_size)

        # Find maximum frequency
        fmax_bin = np.argmax(self.cfft_data)
        self.fmax = self.bin_to_freq(fmax_bin)

        # Power Thresholding
        # if self.cfft_data[vmax_bin] < POWER_THRESHOLD:
        #     self.fmax = 0
        # else:
        #     self.fmax = self.bin_to_freq(vmax_bin)

        self.drho = self.freq_to_vel(self.fmax)
        print('(radar.py) radar', self.index, 'drho:', self.drho)

    def reset(self):
        """Reset all radar data."""
        print("(radar.py) Resetting radar {:}...".format(self.index))
        self.ts_data.clear()


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

            self.last_sample_index = sample_index
        else:
            print('(radar.py) last_sample_index:', self.last_sample_index)
            warning('(radar.py) ignoring stale data...')

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
            return self.radars[self.idx - 1]
        else:
            self.idx = 0
            raise StopIteration()
