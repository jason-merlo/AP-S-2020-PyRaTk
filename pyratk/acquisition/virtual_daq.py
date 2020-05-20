# -*- coding: utf-8 -*-
"""
Virtual DAQ Class.

Description:
DAQ emulation class to playback recorded DAQ datasets in a transparent manner.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
from pyratk.acquisition import daq   # Extention of DAQ object
import threading                     # Used for creating thread and sync events
import time
import h5py
from pyratk.datatypes.ts_data import TimeSeries
from pyratk.datatypes.motion import StateMatrix
from pyqtgraph import QtCore


class VirtualDAQ(daq.DAQ):
    """Emulate DAQ using HDF5 dataset data."""

    def __init__(self):
        """Create virtual DAQ object to play back recording (hdf5 dataset)."""
        super().__init__()

    def load_dataset(self, ds):
        """Select dataset to read from and loads attributes."""
        if isinstance(ds, h5py._hl.dataset.Dataset):
            self.ds = ds
            # self.reset()
        else:
            raise(TypeError,
                  "load_dataset expects a h5py dataset type, got", type(ds))

        # Load attributes
        self.sample_rate = ds.attrs["sample_rate"]
        self.sample_chunk_size = ds.attrs["sample_size"]
        self.daq_type = ds.attrs["daq_type"].decode('utf-8')
        self.num_channels = ds.attrs["num_channels"]
        self.sample_period = self.sample_chunk_size / self.sample_rate

        # Create data buffers
        length = 4096
        shape = (self.num_channels, self.sample_chunk_size)
        self.ts_buffer = TimeSeries(length, shape)

    def load_trajectory(self, ts):
        """Load trajectory dataset."""
        # Trajectory dataset
        self.ts = ts

        # Create data buffers
        length = 4096
        shape = (3, 3)  # State matrix shape
        self.ts_trajectory = TimeSeries(length, shape)


    def get_samples(self, stride=1, loop=-1, playback_speed=1.0):
        """Read sample from dataset at sampled speed, or one-by-one."""
        if self.ds:
            # Read in samples from dataset
            try:
                self.data = self.ds[self.sample_index]
            except IndexError:
                print("Invalid sample index:", self.sample_index)

            if self.ts:
                self._append_trajectory(self.sample_index)

            # Delay by sample period
            if loop == -1 or loop == 1:
                time.sleep(self.sample_period * playback_speed)
            elif loop == 0:
                print('Stepped:', stride)
            else:
                raise ValueError("Value must be -1, 0, or 1.")

            # Append tarjectory before emitting new data signal
            if self.ts:
                self.ts_trajectory.append(self.trajectory_data)

            new_data = (self.data, self.sample_index)
            # Set the update event to True once data is read in
            self.data_available_signal.emit(new_data)
            self.ts_buffer.append(self.data)

            # Incriment time index and loop around at end of dataset
            next_index = self.sample_index + stride
            if next_index < self.ds.shape[0]:
                self.sample_index = next_index
            else:
                self.sample_index = 0
                if self.ts:
                    self.ts_trajectory.clear()
                    self._append_trajectory(self.sample_index)
                self.reset_signal.emit()

            # Return True if more data
            return (self.sample_index + stride) % self.ds.shape[0] / stride < 1.0
        else:
            raise RuntimeError(
                "(VirtualDAQ) Dataset source must be set to get samples")

    def _append_trajectory(self, index):
        coordinate_type = self.ts.attrs['coordinate_type'].decode('utf-8')

        try:
            data = StateMatrix(self.ts[..., self.sample_index * self.sample_chunk_size],
                               coordinate_type=coordinate_type)
        except IndexError:
            print("Invalid trajectory sample index:", self.sample_index)

        self.trajectory_data = data.get_state().q
        self.ts_trajectory.append(self.trajectory_data)

    def reset(self):
        """Reset all data to beginning of data file and begin playing."""
        self.close()
        if self.ts:
            self.ts_trajectory.clear()
            self._append_trajectory(self.sample_index)
        self.sample_index = 0
        self.run()
