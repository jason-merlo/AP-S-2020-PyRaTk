# -*- coding: utf-8 -*-
"""
Measurement Computing DAQ interface class for Windows.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)

Dependencies: mcculw
"""
from pyratk.acquisition.daq import DAQ

# Used for Measurement Computing hardware
from mcculw import ul
from mcculw.enums import (ScanOptions, Status, FunctionType, EventType,
                          TrigType, InfoType, BoardInfo, ULRange, ErrorCode,
                          InterfaceType)
from mcculw.ul import ULError

import ctypes
from collections import namedtuple
from datetime import datetime


class mcdaq_win(DAQ):
    """Measurement Computing DAQ Class for Windows."""

    def __init__(self, sample_rate=50000, sample_chunk_size=5000,
                 low_channel=0, high_channel=7):

        self.daq_type = "MC-DAQ-WIN"

        self._low_channel = low_channel
        self._high_channel = high_channel

        self._board_num = 0
        self._pretrig_count = 0

        # Get available devices
        dev_list = ul.get_daq_device_inventory(InterfaceType.ANY)
        print("Available Devices:", dev_list)

        if dev_list == []:
            raise Exception('No DAQ devices found')

        print(self._get_available_ranges())
        try:
            self._range = self._get_available_ranges()[-1]
        except IndexError:
            self._range = -1

        print('Using range: ', self._range)

        channel_count = self._high_channel - self._low_channel + 1
        self._buffer_size = sample_chunk_size * channel_count

        super().__init__(sample_rate, sample_chunk_size, channel_count)

        self._memhandle = ul.scaled_win_buf_alloc(self._buffer_size)
        self._ctypes_array = ctypes.cast(self._memhandle,
                                         ctypes.POINTER(ctypes.c_double))

        self._scan_options = (ScanOptions.BACKGROUND #| ScanOptions.CONTINUOUS
                              | ScanOptions.SCALEDATA | ScanOptions.EXTTRIGGER
                              | ScanOptions.BURSTIO)

        self._event_types = (EventType.ON_DATA_AVAILABLE
                             | EventType.ON_END_OF_INPUT_SCAN
                             | EventType.ON_SCAN_ERROR)

        self._trig_types = (TrigType.TRIG_POS_EDGE)


        self._last_index = 0

        # Create new sampling task
        # try:
        #     pass
        # except Exception as e:
        #     self.daq_type = "FakeDAQ"
        #     print("="*80)
        #     print("Warning: Exception occurred opening DAQ. Using fake data.")
        #     print(e)
        #     print("="*80)

    def start(self):
        self.run()

    def run(self):
        # Stop any AI task running
        ul.stop_background(self._board_num, FunctionType.AIFUNCTION)

        # Enable the event to be notified every time samples are available.
        print('sample_chunk_size:', self.sample_chunk_size)
        print('self.num_channels:', self.num_channels)
        available_sample_count = self.sample_chunk_size
        print('available_sample_count: ', available_sample_count)

        self._ul_callback = ul.ULEventCallback(self._sample_callback)

        ul.enable_event(self._board_num, self._event_types,
                        self._buffer_size,
                        self._ul_callback, self._ctypes_array)

        # Start the acquisition.
        try:

            # Configure triggering
            ul.set_trigger(self._board_num, self._trig_types,
                0, 0)

            # Run the scan
            ul.a_in_scan(
                self._board_num, self._low_channel, self._high_channel,
                self._buffer_size, self.sample_rate, self._range,
                self._memhandle, self._scan_options)

            # Initialize last sample
            self._last_index = 0
            self._last_time = datetime.now()
            self.paused = False
        except Exception as e:
            print("Error encountered while starting sampling: ", e)

    def _sample_callback(self, board_num, event_type, event_data, c_user_data):
        """Read device sample buffers returning the specified sample size."""

        if (event_type == EventType.ON_DATA_AVAILABLE
                or event_type == EventType.ON_END_OF_INPUT_SCAN):

            # current_time = datetime.now()
            # print('\n')
            # print('delta-time =', current_time - self._last_time)
            # self._last_time = current_time

            chan_count = self._high_channel - self._low_channel + 1

            # print('scan_count: ', scan_count, '\n')
            # print('actual scan rate = ', '{:.6f}'.format(self.sample_rate), 'Hz\n')

            for i in range(chan_count):
                self.data[i, :] = self._ctypes_array[i:self._buffer_size:chan_count]

            new_data = (self.data, self.sample_num)

            # Set the update event to True once data is read in
            self.data_available_signal.emit(new_data)
            self.ts_buffer.append(self.data)

            # Incriment sample number
            self.sample_num += 1

            ul.stop_background(self._board_num, FunctionType.AIFUNCTION)
            ul.a_in_scan(
                self._board_num, self._low_channel, self._high_channel,
                self._buffer_size, self.sample_rate, self._range,
                self._memhandle, self._scan_options)

        elif event_type == EventType.ON_SCAN_ERROR:
            exception = ULException(event_data)
            print(exception)

        elif event_type == EventType.ON_END_OF_INPUT_SCAN:
            print('\nThe scan is complete\n')

    def close(self):
        ul.stop_background(self._board_num, FunctionType.AIFUNCTION)
        super().close()

    def _get_available_ranges(self):
        result = []

        # Check if the board has a switch-selectable, or only one, range
        hard_range = ul.get_config(
            InfoType.BOARDINFO, self._board_num, 0, BoardInfo.RANGE)

        if hard_range >= 0:
            result.append(ULRange(hard_range))
        else:
            for ai_range in ULRange:
                try:
                    ul.a_in(self._board_num, 0, ai_range)
                    result.append(ai_range)
                except ULError as e:
                    if (e.errorcode == ErrorCode.NETDEVINUSE or
                            e.errorcode == ErrorCode.NETDEVINUSEBYANOTHERPROC):
                        raise

        return result
