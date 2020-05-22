# -*- coding: utf-8 -*-
"""
Measurement Computing DAQ interface class.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)

Dependencies: uldaq
"""
from pyratk.acquisition.daq import DAQ

# Used for Measurement Computing hardware
from uldaq import (get_daq_device_inventory, DaqDevice, AInScanFlag,
               DaqEventType, ScanOption, InterfaceType, AiInputMode,
               create_float_buffer, ULException, EventCallbackArgs)

from collections import namedtuple
from datetime import datetime


class mcdaq(DAQ):
    """Measurement Computing DAQ Class."""

    def __init__(self, sample_rate=50000, sample_chunk_size=5000,):

        self.daq_type = "MC-DAQ"

        self._daq_device = None
        self._ai_device = None

        self._descriptor_index = 0
        self._range_index = 3
        self._interface_type = InterfaceType.USB
        self._low_channel = 0
        self._high_channel = 7

        self._event_types = (DaqEventType.ON_DATA_AVAILABLE
                             | DaqEventType.ON_END_OF_INPUT_SCAN
                             | DaqEventType.ON_INPUT_SCAN_ERROR)

        self._scan_params = namedtuple('scan_params',
                                       'buffer high_chan low_chan descriptor status')

        self._scan_options = ScanOption.CONTINUOUS | ScanOption.DEFAULTIO

        self._daq_flags = AInScanFlag.DEFAULT

        # buffer_size = int(2**15 / (self._high_channel - self._low_channel + 1))
        buffer_size = sample_chunk_size

        # Display warning if buffer size < 2*chunk_size
        if buffer_size < 2 * sample_chunk_size:
            print('Warning: Buffer size < 2 * sample_chunk_size')

        channel_count = self._high_channel - self._low_channel + 1
        super().__init__(sample_rate, buffer_size, channel_count)

        # Initialize last sample
        self._last_index = 0

        # Create new sampling task
        try:
            # Get descriptors for all of the available DAQ devices.
            devices = get_daq_device_inventory(self._interface_type)
            number_of_devices = len(devices)
            if number_of_devices == 0:
                raise Exception('Error: No DAQ devices found')

            print('Found', number_of_devices, 'DAQ device(s):')
            for i in range(number_of_devices):
                print('  ', devices[i].product_name, ' (', devices[i].unique_id, ')', sep='')

            # Create the DAQ device object associated with the specified descriptor index.
            self._daq_device = DaqDevice(devices[self._descriptor_index])

            # Get the AiDevice object and verify that it is valid.
            self._ai_device = self._daq_device.get_ai_device()
            if self._ai_device is None:
                raise Exception('Error: The DAQ device does not support analog input')

            # Verify that the specified device supports hardware pacing for analog input.
            self._ai_info = self._ai_device.get_info()
            if not self._ai_info.has_pacer():
                raise Exception('\nError: The specified DAQ device does not support hardware paced analog input')

            # Establish a connection to the DAQ device.
            self._descriptor = self._daq_device.get_descriptor()
            print('\nConnecting to', self._descriptor.dev_string, '- please wait...')
            self._daq_device.connect()

            # The default input mode is SINGLE_ENDED.
            self._input_mode = AiInputMode.SINGLE_ENDED
            # If SINGLE_ENDED input mode is not supported, set to DIFFERENTIAL.
            if self._ai_info.get_num_chans_by_mode(AiInputMode.SINGLE_ENDED) <= 0:
                self._input_mode = AiInputMode.DIFFERENTIAL

            # Get a list of supported ranges and validate the range index.
            ranges = self._ai_info.get_ranges(self._input_mode)
            if self._range_index >= len(ranges):
                self._range_index = len(ranges) - 1

            self._range = ranges[self._range_index]

            # Allocate a buffer to receive the data
            self._data = create_float_buffer(self.num_channels,
                                             self.sample_chunk_size)

            # Store the user data for use in the callback function.
            self._scan_status = {'complete': False, 'error': False}
            self._user_data = self._scan_params(self._data,
                                                self._high_channel,
                                                self._low_channel,
                                                self._descriptor,
                                                self._scan_status)

            print('\n', self._descriptor.dev_string, ' ready', sep='')
            print('    Function demonstrated: self._ai_device.a_in_scan()')
            print('    Channels: ', self._low_channel, '-', self._high_channel)
            print('    Input mode: ', self._input_mode.name)
            print('    Range: ', self._range.name)
            print('    Samples per channel: ', self.sample_chunk_size)
            print('    Rate: ', self.sample_rate, 'Hz')
            print('    Scan options:', self._display_scan_options(self._scan_options))

        except Exception as e:
            self.daq_type = "FakeDAQ"
            print("="*80)
            print("Warning: Exception occurred opening DAQ. Using fake data.")
            print(e)
            print("="*80)

    def run(self):
        # Enable the event to be notified every time samples are available.
        print('sample_chunk_size:', self.sample_chunk_size)
        print('self.num_channels:', self.num_channels)
        available_sample_count = self.sample_chunk_size
        print('available_sample_count: ', available_sample_count)
        self._daq_device.enable_event(self._event_types, available_sample_count,
                                self._sample_callback, self._user_data)

        # Start the acquisition.
        try:
            rate = self._ai_device.a_in_scan(self._low_channel, self._high_channel,
                                             self._input_mode, self._range,
                                             self.sample_chunk_size, self.sample_rate,
                                             self._scan_options, self._daq_flags,
                                             self._data)

            # Initialize last sample
            self._last_index = 0
            self._last_time = datetime.now()

        except Exception as e:
            print("Error encountered while starting sampling: ", e)

    def _sample_callback(self, event_callback_args):
        """Read device sample buffers returning the specified sample size."""

        event_type = DaqEventType(event_callback_args.event_type)
        scan_count = event_callback_args.event_data
        user_data = event_callback_args.user_data

        if (event_type == DaqEventType.ON_DATA_AVAILABLE
                or event_type == DaqEventType.ON_END_OF_INPUT_SCAN):

            # current_time = datetime.now()
            # print('\n'*2, 'delta-time =', current_time - self._last_time)
            # self._last_time = current_time

            chan_count = user_data.high_chan - user_data.low_chan + 1
            total_samples = scan_count * chan_count

            # print('scan_count: ', scan_count, '\n')
            # print('actual scan rate = ', '{:.6f}'.format(self.sample_rate), 'Hz\n')

            for i in range(chan_count):
                self.data[i, :] = user_data.buffer[i::chan_count]

        if event_type == DaqEventType.ON_INPUT_SCAN_ERROR:
            exception = ULException(event_data)
            print(exception)
            user_data.status['error'] = True

        if event_type == DaqEventType.ON_END_OF_INPUT_SCAN:
            print('\nThe scan is complete\n')
            user_data.status['complete'] = True

    def start(self):
        self.run()

    def close(self):
        if self._daq_device:
            if self._daq_device.is_connected():
                # Stop the acquisition if it is still running.
                if self._ai_device and self._ai_info and self._ai_info.has_pacer():
                    self._ai_device.scan_stop()
                self._daq_device.disable_event(self._event_types)
                self._daq_device.disconnect()
            self._daq_device.release()

        super().close()

    def _display_scan_options(self, bit_mask):
        options = []
        if bit_mask == ScanOption.DEFAULTIO:
            options.append(ScanOption.DEFAULTIO.name)
        for so in ScanOption:
            if so & bit_mask:
                options.append(so.name)
        return ', '.join(options)
