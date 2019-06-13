# -*- coding: utf-8 -*-
"""
DAQ Manager class.

Handles interfacing with physical DAQ and controls data flow from DAQ.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)

Dependencies: nidaqmx, random, threading, time, numpy
"""
try:
    import nidaqmx              # Used for NI-DAQ hardware
    from nidaqmx import stream_readers
except ImportError:
    print('Warning: nidaqmx module not imported')
import threading                # Used for creating thread and sync events
import time
import numpy as np
from ts_data import TimeSeries


class DAQ(object):
    def __init__(self, daq_type="nidaq",
                 sample_rate=44100, sample_size=4096,
                 # NI-DAQ specific
                 dev_string="Dev1/ai0:7",
                 sample_mode=nidaqmx.constants.AcquisitionType.FINITE):
        """
        Create sampling task on DAQ and opens I & Q channels for radars.

        arguments:
        dev_string -- device and ports to initialize (default: "Dev1/ai0:7")
        sample_rate -- frequency in Hz to sample at (default: 44100)
        sample_mode -- finite or continuous acquisition (default: finite)
        sample_size -- size of chunk to read (default/max: 4095)
        """
        # Copy member data
        # General arguments
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.daq_type = daq_type
        self.paused = False  # Start running by default
        self.reset_flag = False

        # Create sevent for controlling draw events only when there is new data
        self.data_available = threading.Event()

        self.sample_num = 0

        # Device specific arguments
        if self.daq_type == "nidaq":
            self.sample_mode = sample_mode
            self.dev_string = dev_string
            # Get number of channels to sample
            if self.dev_string[-2] == ':':
                self.num_channels = int(
                    self.dev_string[-1]) - int(self.dev_string[-3]) + 1
            else:
                self.num_channels = int(self.dev_string[-1]) + 1

            # Create new sampling task
            try:
                # Try to create sampling task
                self.task = nidaqmx.Task()

                self.task.ai_channels.add_ai_voltage_chan(dev_string)

                self.task.timing.cfg_samp_clk_timing(
                    sample_rate, sample_mode=sample_mode,
                    samps_per_chan=sample_size)
                self.in_stream = \
                    stream_readers.AnalogMultiChannelReader(
                        self.task.in_stream)
            except nidaqmx._lib.DaqNotFoundError:
                # On failure (ex. on mac/linux) generate random data for
                # development purposes
                # TODO: switch to PyDAQmx for mac/linux
                # TODO: is there any reason to keep nidaqmx for windows?
                # TODO: try performance comparison
                self.daq_type = "fake_daq"
                print("="*60)
                print("Warning: Using fake data. nidaqmx is not "
                      "supported on this platform.")
                print("="*60)
            except nidaqmx.errors.DaqError as e:
                print(e)
                self.daq_type = "fake_daq"
                print("="*60)
                print("Warning: Using fake data. DAQ could not be detected.")
                print("="*60)

        elif self.daq_type == "pyaudio":
            pass  # TODO insert pyaudio support here

        # Create data member to store samples
        self.data = np.empty((self.num_channels, self.sample_size),)

        # Spawn sampling thread
        self.run()

        # Create data buffers
        self.buffer = []
        length = 4096
        shape = (self.num_channels, self.sample_size)
        self.ts_buffer = TimeSeries(length, shape)

    # === SAMPLING ======================================================
    def sample_loop(self):
        """
        Calls get_samples forever
        """
        while self.running:
            if self.paused:
                time.sleep(0.1)  # sleep 100 ms
            else:
                self.get_samples()
        print("Sampling thread stopped.")

    def get_samples(self):
        """
        Reads device sample buffers returning the specified sample size

        arguments:
        task -- nidaqmx task object, returned from open_task_channels()
        """
        # print("DAQ updated...")

        if self.daq_type == "fake_daq":
            sleep_time = self.sample_size / self.sample_rate
            self.data = np.random.randn(
                self.num_channels, self.sample_size) * 0.001 + \
                np.random.randn(1) * 0.001 + 0.01
            time.sleep(sleep_time)

            # print('received update')
        else:
            try:
                read_all = nidaqmx.constants.READ_ALL_AVAILABLE
                self.in_stream.read_many_sample(
                    self.data,
                    number_of_samples_per_channel=read_all,
                    timeout=1.0)

                # print('received update')
            except nidaqmx.errors.DaqError as err:
                print("DAQ exception caught: {0}\n".format(err))

        self.buffer.append((self.data, self.sample_num))
        self.ts_buffer.append(self.data)

        self.sample_num += 1
        # Set the update event to True once data is read in
        self.data_available.set()

    # === CONTROL =======================================================
    def close(self):
        print("Stopping sampling thread...")
        self.running = False
        if self.daq_type == "nidaq" and hasattr(self, 'task'):
            self.task.close()  # Close nidaq gracefully
        if self.t_sampling.is_alive():
            try:
                self.t_sampling.join()
            except Exception as e:
                print("Error closing sampling thread: ", e)

    def run(self):
        # Spawn sampling thread
        self.running = True
        self.t_sampling = threading.Thread(target=self.sample_loop)
        try:
            if not self.t_sampling.is_alive():
                print('Staring sampling thread')
                self.t_sampling.start()
            self.paused = False
        except RuntimeError as e:
            print('Error starting sampling thread: ', e)

    def pause(self):
        self.paused = True

    def reset(self):
        self.buffer = []
        self.ts_buffer.clear()
        self.sample_num = 0
        self.reset_flag = True

    # === PROPERTIES ====================================================
    @property
    def type(self):
        return self.daq_type