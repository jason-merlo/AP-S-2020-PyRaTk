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
from scipy import signal        # Upsampling, bh window
from pyratk.datatypes.geometry import Point         # radar location
# from pyratk.datatypes.motion import StateMatrix   # track location
from pyqtgraph import QtCore

from pyratk.formatting import warning

from profilehooks import profile

from collections import namedtuple

class Receiver(object):
    """
    Object to hold Radar Receiver information and signal processing.

    It is assumed radars will have both I and Q channels.

    NOTE: Radar class should not contain state history (TimeSeries).
          Work on removing these and placing them in Tracker or DataLogger.

    """

    # === INITIALIZATION METHODS ==============================================
    def __init__(self, daq, daq_index, transmitter, loc=Point(),
                 fast_fft_size=2**12, fast_fft_window_type=None,
                 slow_fft_size=2**12, slow_fft_window_type=None,
                 slow_fft_len=20):
        super().__init__()
        """
        Initialize radar object.

        Note: Currently only FMCW or CW radars are supported.

        Args:
            daq
                [daq] object for DAQ configurations
            daq_index
                Index of radar I & Q channels in DAQ output
            transmitter
                Transmitter object
            loc
                Point() indicating the location of the radar

        """
        # Data stream parameters
        # TODO: create DataStream object representing I or I/Q data
        self.daq = daq
        self.daq_index = daq_index

        # Physical parameters
        self.transmitter = transmitter
        self.loc = loc

        # Processing parameters
        self.fast_fft_size = fast_fft_size
        self.fast_fft_window_type = fast_fft_window_type
        self.slow_fft_size = slow_fft_size
        self.slow_fft_window_type = slow_fft_window_type
        self.slow_fft_len = slow_fft_len

        self.init_data()
        self.connect_signals()

    def init_data(self):
        # Derived processing parameters
        self.fast_center_bin = np.ceil(self.fast_fft_size / 2)
        self.fast_bin_size = self.daq.sample_rate / self.fast_fft_size
        self.slow_center_bin = np.ceil(self.slow_fft_size / 2)
        self.slow_bin_size = self.daq.sample_rate / self.slow_fft_size

    def connect_signals(self):
        # self.daq.reset_signal.connect(self.reset)
        pass


    # === HELPER METHODS ======================================================
    def freq_to_vel(self, freq):
        """Compute the velocity for a given frequency and the radar fc."""
        c = spc.speed_of_light
        # velocity = (c * self.fc / (freq + self.fc)) - c
        velocity = (freq * c) / (2 * self.fc)

        return velocity

    def bin_to_freq(self, bin):
        """Compute frequency based on bin location."""
        return (bin - self.center_bin) * float(self.bin_size)

    def compute_cfft(self, complex_data, fft_size):
        """Compute fft and fft magnitude for plotting."""
        # Create blackman-harris window
        # window = signal.blackmanharris(complex_data.shape[0])
        # window = signal.hamming(complex_data.shape[0])
        window = np.ones(complex_data.shape[0])

        # Create zero-padded array to be transformed
        fft_array = np.zeros((fft_size,), dtype=np.complex64)
        fft_array[:complex_data.size] = complex_data * window

        # Normalize FFT magnitude to window size
        fft_complex = np.fft.fft(fft_array, norm='ortho')

        # Adjust fft so DC is at the center
        fft_complex = np.fft.fftshift(fft_complex)

        # Display only magnitude
        fft_mag = np.linalg.norm([fft_complex.real, fft_complex.imag], axis=0)

        return fft_mag

    # === CONTROL METHODS =====================================================
    def update(self, data):
        # Get data from daq
        data_slice = np.array((data[self.daq_index[0]], data[self.daq_index[1]]))

        # self.data_buffer = np.append(self.data_buffer, iq_data_slice)

        # Get window of FFT data
        window_idx = self.window_size // self.daq.sample_chunk_size
        window_slice_pair = self.ts_data[-window_idx:, ...]
        window_slice = window_slice_pair[:, 0, :].flatten() \
            + window_slice_pair[:, 1, :].flatten() * 1.0j
        # window_slice = window_slice_pair[:, 1, :].flatten()

        window_slice -= np.mean(window_slice.real) + np.mean(window_slice.imag) * 1.0j

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

    def reset(self):
        """Reset all radar data."""
        self.init_data()

    @property
    def datacube(frames):
        """
        Create datacube with specified number of most recent frames.

        Returns a complex datacube.  Final shape will be:
        frames x slow_fft_len x samples_per_chirp
        """
        idx = self.daq_index
        samples = self.daq.sample_rate = self.transmitter.pulses[0].delay
        return self.daq.ts_buffer[-frames:, idx[0], samples] \
               + 1.0j * self.daq.ts_buffer[-frames:, idx[1], samples]


class Transmitter(object):
    """
    Object to hold Radar Transmitter information.

    Note: This class may perform transmitter control functions in the future.
    """

    def __init__(self, data_mgr, pulses, loc=Point()):
        super().__init__()
        """
        Initialize radar object.

        Note: Currently only FMCW or CW radars are supported.

        Args:
            data_mgr
                [data_mgr] object for DAQ configurations
            loc
                Point() indicating the location of the radar
            pulses
                List of pulse namedtuple objects containing fc, bw, and delay

                fc
                    Center frequency of transmit waveform
                bw
                    Bandwidth of transmit chirp waveform
                delay
                    Delay time between start of chirp

        """
        self.data_mgr = data_mgr
        self.pulses = pulses

        # Currently only one pulse is supported due to datacube restrictions
        if len(self.pulses) != 1:
            raise NotImplementedError('Only one pulse segment is currently'
            ' supported due to datacube processing restrictions')


class Radar(QtCore.QObject):
    """
    Holds radar Transmitter and Receiver objects.

    Note: all iterable functions still only iterate over receiver for
    backwards compatibility with old RadarArray structure.

    Attributes:
        tranmitters (namedtuple)
            location (Point)
            pulses (namedtuple)
                fc
                    Pulse waveform center frequency
                bw
                    Pulse waveform bandwidth
                delay
                    Time between start of waveform modulation

        receivers (namedtuple)
            daq_index (tuple)
                tuple containing index of I and Q channels on DAQ
            location (Point)

        fast_fft_size
        fast_fft_window_type
        slow_fft_size
        slow_fft_window_type
        slow_fft_len

    """
    data_available_signal = QtCore.pyqtSignal()

    def __init__(self,
        daq,
        transmitter_list,
        receiver_list,
        fast_fft_size=2**10,
        fast_fft_window_type=None,
        slow_fft_size=2**12,
        slow_fft_window_type=None,
        slow_fft_len=100
    ):
        super().__init__()

        self.daq = daq

        # Used for iterator magic functions
        self.idx = 0

        # Flag to iqnore any stale data after a reset
        self.last_sample_index = -1

        # Create recievers and transmitters
        self.transmitters = []
        self.receivers = []

        for transmitter in transmitter_list:
            self.transmitters.append(
                Transmitter(
                    self.daq,
                    transmitter.pulses
                )
            )

        for receiver in receiver_list:
            self.receivers.append(
                Receiver(
                    self.daq,
                    receiver.daq_index,
                    self.transmitters[0],
                    receiver.location,
                    fast_fft_size,
                    fast_fft_window_type,
                    slow_fft_size,
                    slow_fft_window_type,
                    slow_fft_len
                )
            )

        # Configure Signals
        self.connect_signals()

    def connect_signals(self):
        """Connect signals for event driven operation."""
        self.daq.data_available_signal.connect(self.update)
        self.daq.reset_signal.connect(self.reset)

    def reset(self):
        """Clear all temporal data from radars in array."""
        for receiver in self.receivers:
            receiver.reset()
        self.last_sample_index = -1

    # @profile(immediate=True)
    def update(self, data_tuple):
        """Update all radars in array."""
        # start_time = time.time()
        data, sample_index = data_tuple

        # print('(radar.py) sample_num:', sample_index)

        if sample_index == self.last_sample_index + 1:
            self.reset_flag = False
            for receiver in self.receivers:
                receiver.update(data)

            # Emit data available signal for dependant tasks
            self.data_available_signal.emit()

            self.last_sample_index = sample_index
        else:
            # print('(radar.py) last_sample_index:', self.last_sample_index)
            # warning('(radar.py) ignoring stale data...')
            pass

        # print('(radar.py) radar_array.update() ran in {:} (s)'
        #       .format(time.time() - start_time))

    def __getitem__(self, i):
        """Return radar in array."""
        assert(i < len(self)), "index greater than flattened array size"
        return self.receivers[i]

    def __len__(self):
        """Return length of radar array."""
        # compute sumproduct of array shape
        return len(self.receivers)

    def __iter__(self):
        """Return self for iterator."""
        return self

    def __next__(self):
        """Return next radar object for iterator."""
        if self.idx < len(self):
            self.idx += 1
            return self.receivers[self.idx - 1]
        else:
            self.idx = 0
            raise StopIteration()
