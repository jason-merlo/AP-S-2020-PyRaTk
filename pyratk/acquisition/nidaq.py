# -*- coding: utf-8 -*-
"""
National Instruments DAQ interface class wrapper.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)

Dependencies: uldaq
"""
from pyratk.acquisition.daq import DAQ

try:
    import nidaqmx              # Used for National Instruments hardware
except ImportError:
    print('Warning: nidaqmx module not imported')

class nidaq(DAQ):
    """National Instruments DAQ Class."""

    def __init__(self, sample_rate=44100, sample_chunk_size=4096,
                 ni_dev_string="Dev1/ai0:7",
                 ni_sample_mode=nidaqmx.constants.AcquisitionType.FINITE,):
        """
        Initialize NI-DAQ class.

        arguments:
        sample_rate -- frequency in Hz to sample at (default: 44100)
        sample_chunk_size -- size of chunk to read (default/max: 4095)
        ni_dev_string -- device and ports to initialize (default: "Dev1/ai0:7")
        ni_sample_mode -- finite or continuous acquisition (default: finite)
        """
        self.daq_type='NI-DAQ'

        self.ni_sample_mode = ni_sample_mode
        self.ni_dev_string = ni_dev_string
        # Get number of channels to sample
        if self.ni_dev_string[-2] == ':':
            num_channels = int(
                self.ni_dev_string[-1]) - int(self.ni_dev_string[-3]) + 1
        else:
            num_channels = int(self.ni_dev_string[-1]) + 1


        super().__init__(sample_rate, sample_chunk_size, num_channels)

        # Create new sampling task
        try:
            # Try to create sampling task
            self.task = nidaqmx.Task()

            self.task.ai_channels.add_ai_voltage_chan(ni_dev_string)

            self.task.timing.cfg_samp_clk_timing(
                sample_rate, ni_sample_mode=ni_sample_mode,
                samps_per_chan=sample_chunk_size)
            self.in_stream = \
                stream_readers.AnalogMultiChannelReader(
                    self.task.in_stream)
        except nidaqmx._lib.DaqNotFoundError:
            # On failure (ex. on mac/linux) generate random data for
            # development purposes
            # TODO: switch to PyDAQmx for mac/linux
            # TODO: is there any reason to keep nidaqmx for windows?
            # TODO: try performance comparison
            self.daq_type = "FakeDAQ"
            print("="*80)
            print("Warning: Using fake data. nidaqmx is not "
                  "supported on this platform.")
            print("="*80)
        except nidaqmx.errors.DaqError as e:
            print(e)
            self.daq_type = "FakeDAQ"
            print("="*80)
            print("Warning: Using fake data. DAQ could not be detected.")
            print("="*80)

    def get_samples(self):
        """Read device sample buffers returning the specified sample size."""

        try:
            read_all = nidaqmx.constants.READ_ALL_AVAILABLE
            self.in_stream.read_many_sample(
                self.data,
                number_of_samples_per_channel=read_all,
                timeout=1.0)

            # print('received update')
        except nidaqmx.errors.DaqError as err:
            print("DAQ exception caught: {0}\n".format(err))

    # === SAMPLING ======================================================
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

    def get_fake_samples(self):
        """Generate fake DAQ samples."""
        sleep_time = self.sample_chunk_size / self.sample_rate
        self.data = np.random.randn(
            self.num_channels, self.sample_chunk_size) * 0.001 + \
            np.random.randn(1) * 0.001 + 0.01
        time.sleep(sleep_time)


        def run(self):
            if self.running == False:
                # Spawn sampling thread
                self.running = True
                self.t_sampling = threading.Thread(target=self.sample_loop)

                # Trigger DAQ to start running
                self.start()

                try:
                    if not self.t_sampling.is_alive():
                        print('Staring sampling thread')
                        self.t_sampling.start()
                    self.paused = False
                except RuntimeError as e:
                    print('Error starting sampling thread: ', e)
            else:
                print('Warning: Not starting new sampling thread; sampling thread already running!')

    def start(self):
        self.run()

    def close(self):
        self.task.close()  # Close nidaq gracefully

        if self.t_sampling.is_alive():

            print("Stopping sampling thread...")
            try:
                self.t_sampling.join()
            except Exception as e:
                print("Error closing sampling thread: ", e)
                
        super().close()
