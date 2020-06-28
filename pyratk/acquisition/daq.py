# -*- coding: utf-8 -*-
"""
DAQ Manager class.

Handles interfacing with physical DAQ and controls data flow from DAQ.

TODO: Abstract base DAQ class from NI-DAQ class.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)

Dependencies: nidaqmx, random, Threading, time, numpy
"""
import threading                # Used for creating thread and sync events
import time
import numpy as np

from pyqtgraph import QtCore
from pyratk.datatypes.ts_data import TimeSeries

from pyratk.formatting import warning


class DAQ(QtCore.QThread):
    # Create event for processing new data only when available
    data_available_signal = QtCore.pyqtSignal(tuple)
    reset_signal = QtCore.pyqtSignal()  # required for VirtualDAQ

    def __init__(self, sample_rate=50000, sample_chunk_size=50000,
                 num_channels=8):
        """
        Create sampling task on DAQ and opens I & Q channels for radars.

        Emits a signal when new data is available.
        """
        super().__init__()
        # Copy member data
        # General arguments
        self.sample_rate = sample_rate
        self.sample_chunk_size = sample_chunk_size
        self.num_channels = num_channels
        self.update_period = sample_chunk_size / sample_rate
        self.paused = True
        self.running = False

        self.sample_num = 0

        # Create data member to store samples
        self.data = np.empty((self.num_channels, self.sample_chunk_size),)

        # Create data buffer for saving data
        length = 4096
        shape = (self.num_channels, self.sample_chunk_size)
        self.ts_buffer = TimeSeries(length, shape)

    def sample_loop(self):
        """Call get_samples forever."""
        while self.running:
            if self.paused:
                # warning('(daq.py) daq paused...')
                time.sleep(0.1)  # sleep 100 ms
            else:
                if self.daq_type == "FakeDAQ":
                    self.get_fake_samples()
                else:
                    self.get_samples()

                new_data = (self.data, self.sample_num)

                # Set the update event to True once data is read in
                self.data_available_signal.emit(new_data)
                self.ts_buffer.append(self.data)

                # Incriment sample number
                self.sample_num += 1

        print("Sampling thread stopped.")

    # === CONTROL =======================================================
    def close(self):
        self.running = False

    def pause(self):
        self.paused = True

    def reset(self):
        self.ts_buffer.clear()
        self.sample_num = 0

    def start(self):
        """
        Begin sampling process on DAQ.

        To be implemented by child classes.
        """
        pass

    # === PROPERTIES ====================================================
    @property
    def type(self):
        return self.daq_type
