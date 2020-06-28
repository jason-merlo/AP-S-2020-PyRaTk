"""
SynthDAQ Class file.

WARNING: SynthDAQ is still very much in development

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import logging                       # Debug messages
import numpy as np                   # Data processing
import threading                     # Asynchronous operation of DAQ
# import time                          # Sleeping DAQ sample thread

import matplotlib.pyplot as plt      # Graphing trajectory
import scipy.constants as spc        # speed of light
# import multiprocess as mp
from multiprocess import Pool
# from pathos.multiprocessing import ProcessingPool as Pool

from pyratk.acquisition import daq   # Base DAQ class
from pyratk.datatypes.motion import StateMatrix  # Tracking state
from pyratk.datatypes.ts_data import TimeSeries  # Storing time-series data
from pyratk.datatypes.geometry import Point      # Coodinate systems


# Configure logging options
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class SynthDAQ(daq.DAQ):
    """Generate synthetic datasets based on parametric trajectories."""

    coordinate_types = {
        'cartesian': ('x', 'y', 'z'),
        'cylindrical': ('r', 'theta', 'z'),
        'spherical': ('rho', 'theta', 'phi')
    }

    def __init__(self, daq, array):
        """Create SynthDAQ to generate synthetic data.

        Args:
            array (type): Description of parameter `array`.
            daq (type): Description of parameter `daq`.
            trajectory_samples_per_sample (type): Description of parameter `trajectory_samples_per_sample`.

        Returns:
            None: no return.

        """
        super().__init__()
        # DAQ Attributes
        self.sample_interval = daq['sample_interval']
        self.sample_rate = 1.0 / daq['sample_interval']
        self.sample_chunk_size = daq['sample_size']
        self.daq_type = 'SynthDAQ'
        self.num_channels = 2 * len(array['radar_list'])

        # Array Attributes
        self.array = array['radar_list']

        # start paused if no dataset is selected
        self.paused = True

        # Allow points to be generated as fast as possible
        self.time_warp = True

        self.reset_flag = False

        # Fake sampler period
        self.sample_chunk_period = self.sample_period = self.sample_chunk_size / self.sample_rate

        # Create data member to store samples
        self.data = None
        self.ds = None

        # Current time index of recording
        self.sample_index = 0

        # Create sevent for controlling draw events only when there is new data
        self.data_available = threading.Event()

        # Reset/load button sample thread locking
        self.reset_lock = threading.Event()

        self.t_sampling = threading.Thread(target=self.sample_loop)

        # Buffer for sampled data prior to processing
        self.buffer = []
        length = 4096
        shape = (self.num_channels, self.sample_chunk_size)
        self.ts_buffer = TimeSeries(length, shape)

        # === Trajectory variables === #
        # Placeholder for trajectory description
        self.coordinate_type = None
        self.waypoints = []
        self.trajectory_samples = np.zeros((3, 3, 0))
        self.radar_samples = []

    def load_trajectory(self, trajectory_dict):
        """Load in trajectory data from dict.

        Args:
            trajectory_dict (type): Dictionary containing trajectory
            information parsed from .ymal file.

        Raises:
            RuntimeError: If required keys are missing from .yaml file.
            RuntimeWarning: If fewer than 2 waypoints are provided for path.

        Returns:
            None: No object returned.

        """
        # === Check for valid parameters === #

        # Check for required members in dict
        missing_keys = set(['coordinate_type',
                            'waypoints']).difference(trajectory_dict.keys())
        if missing_keys:
            raise RuntimeError("Missing required trajectory member:",
                               missing_keys)

        # Check for number of waypoints in file
        if len(trajectory_dict['waypoints']) <= 1:
            raise RuntimeWarning(
                "Number of waypoints should be greater than 1")

        # === Copy in required data === #
        self.coordinate_type = trajectory_dict['coordinate_type']

        # Generate list of waypoint states
        for wp in trajectory_dict['waypoints']:
            q = StateMatrix(np.array(wp)[:, :, 0])
            q_max = StateMatrix(np.array(wp)[:, :, 1])
            self.waypoints.append({'q': q, 'q_max': q_max})

    def generate_trajectory_function(self):
        """
        Compute trajectory function coefficients.

        Pre-compute trajectory funciton coefficiets to allow for arbitrary
        time sampling of target state(s) at a given time.
        """
        pass

    def generate_trajectory_samples(self):
        """
        Pre-compute trajectory and sample location.

        Create array of StateMatrix at locations derived from DAQ sampling
        period using Linear Segments with Parabolic Blends (LSPB).
        """
        # TODO: Do not pre-compute; causes issues for modulated waveforms
        # TODO: Break into smaller functions to reduce cyclonomic complexity.

        # Check if trajectory file has been loaded
        if self.waypoints == []:
            raise RuntimeError("Trajectory description file required to "
                               "generate trajectory samples.  Run "
                               "load_trajectory() first.")

        # Calculate sample period
        logging.debug('sample_rate: {:+6.3f} (Hz)'.
                      format(1.0 / self.sample_interval))

        # Iterate through each pair of waypoints to calculate trajectory
        wp1 = None
        for wp_idx, wp2 in enumerate(self.waypoints):
            # if first loop, copy in wp1 and restart loop
            if wp1 is None:
                wp1 = wp2
                continue

            # Correct waypoint index
            wp_idx -= 1

            # Create arrays to store blend and total times for each axis
            time_param_array = np.zeros((3, 3))

            print('=' * 10 + ' segment: {:} '.format(wp_idx) + '=' * 60)

            logging.debug('Computing time parameters to find longest time')
            logging.debug('wp1:\n{:}'.format(wp1['q']))
            logging.debug('wp2:\n{:}'.format(wp2['q']))

            # Iterate over each spatial dimension
            for dim in range(len(wp1['q'])):
                # Copy in as a and b for ease of reference
                a = wp1['q'][dim]
                b = wp2['q'][dim]
                q_max = wp1['q_max'][dim]

                # Check if any motion occurs in dimension; if not, skip
                if a[0] == b[0]:
                    logging.debug(
                        'No motion on axis {:}, skipping.'.format(dim))
                    continue

                # Flip waypoints and set reverse flag for calculations
                if b[0] < a[0]:
                    logging.debug('Negative motion, swapping waypionts.')
                    for idx in range(len(a) - 1):
                        a[idx], b[idx] = -a[idx], -b[idx]
                    reverse = True
                else:
                    reverse = False

                # Calculate maximum blend times
                tb1_max = abs((q_max[1] - a[1]) / a[2])
                tb2_max = abs((q_max[1] - b[1]) / a[2])

                # Calculate total time using maximum blend times
                tf_max = (b[0] - a[0] - a[1] * tb1_max - q_max[1] * tb2_max
                          - 0.5 * a[2] * (tb1_max**2 - tb2_max**2)
                          ) / q_max[1] + (tb1_max + tb2_max)

                logging.debug('tb1_max: {:+1.3} (s)'.format(tb1_max))
                logging.debug('tb2_max: {:+1.3} (s)'.format(tb2_max))

                logging.debug('tf_max: {:+1.3} (s)'.format(tf_max))

                # If total time must be less than the sum of the blend times,
                # the max velocity is not reached; no linear segment needed.
                if tf_max < tb1_max + tb2_max:
                    # logging.debug(
                    #     'Max. velocity not reached, using two segments.')

                    # Calculate blend and total time using two parabolic
                    # segments
                    R = np.sqrt(2 * (a[1]**2 + b[1]**2
                                     - 2 * a[0] * a[2]
                                     + 2 * a[2] * b[0]))

                    tb1 = -(2 * a[1] - R) / (2 * a[2])

                    tf = (a[1] - b[1]) / a[2] + 2 * tb1

                    # Set final values for blends and total time
                    tb2 = tf - tb1
                else:
                    tb1 = tb1_max
                    tb2 = tb2_max
                    tf = tf_max

                # Add time parameters to list
                time_param_array[dim] = np.array((tb1, tb2, tf))

                # Correct waypoints before continuing
                if reverse:
                    for idx in range(len(a) - 1):
                        a[idx], b[idx] = -a[idx], -b[idx]

            # Choose longest total time, rescale others
            max_dim = np.argmax(time_param_array[:, 2])
            tf = time_param_array[max_dim][2]
            scale_factor = np.ones((3,))

            for dim in range(len(wp1['q'])):
                # Do not re-scale dimension with max time
                if dim == max_dim:
                    continue

                # Skip axis with no total time (no motion)
                if time_param_array[dim][2] == 0:
                    continue

                # Re-scale blends
                scale_factor[dim] = tf / time_param_array[dim][2]
                for idx in range(2):
                    time_param_array[dim][idx] *= scale_factor[dim]

                time_param_array[dim][2] = tf

            logging.debug('max_dim: {:1}'.format(max_dim))
            logging.debug('max_tf: {:+1.3} (s)'.format(tf))

            # Calculate sampling times
            # NOTE: sample trajectory with DAQ sample_rate to ensure
            # re-creation of sample spread.
            num_samples = tf * self.sample_rate
            ts = np.linspace(0, tf, num_samples)  # sampled times

            # Iterate over each spatial dimension
            for dim in range(len(wp1['q'])):
                # Copy in as a and b for ease of reference
                a = wp1['q'][dim]
                b = wp2['q'][dim]
                q_max = wp1['q_max'][dim]

                print('=' * 20 + ' axis: ' + str(dim) + ' ' + '=' * 20)

                tb1 = time_param_array[dim][0]
                tb2 = time_param_array[dim][1]

                # Flip waypoints and set reverse flag for calculations
                if b[0] < a[0]:
                    logging.debug('Negative motion, swapping waypionts.')
                    for idx in range(len(a) - 1):
                        a[idx], b[idx] = -a[idx], -b[idx]
                    reverse = True
                else:
                    reverse = False

                # Create piecewise function for trajectory
                conditions = [
                    ts < tb1,
                    (tb1 <= ts) * (ts < tf - tb2),
                    tf - tb2 <= ts
                ]
                if reverse:
                    functions = [
                        lambda ts: np.array((- a[0] - a[1] * ts - 0.5 * a[2] * ts**2,
                                             - (a[1] + a[2] * ts),
                                             - a[2] + 0 * ts)),
                        lambda ts: np.array((functions[0](tb1)[0] + functions[0](tb1)[1] * (ts - tb1),
                                             functions[0](tb1)[1] + 0 * ts,
                                             0 + 0 * ts)),
                        lambda ts: np.array((functions[1](tf - tb2)[0] + functions[1](tf - tb2)[1] * (ts - (tf - tb2)) + 0.5 * a[2] * (ts - (tf - tb2))**2,
                                             functions[1](
                                                 tf - tb2)[1] + a[2] * (ts - (tf - tb2)),
                                             a[2] + 0 * ts))
                    ]
                else:
                    functions = [
                        lambda ts: np.array((a[0] + a[1] * ts + 0.5 * a[2] * ts**2,
                                             a[1] + a[2] * ts,
                                             a[2] + 0 * ts)),
                        lambda ts: np.array((functions[0](tb1)[0] + functions[0](tb1)[1] * (ts - tb1),
                                             functions[0](tb1)[1] + 0 * ts,
                                             0 + 0 * ts)),
                        lambda ts: np.array((functions[1](tf - tb2)[0] + functions[1](tf - tb2)[1] * (ts - (tf - tb2)) - 0.5 * a[2] * (ts - (tf - tb2))**2,
                                             functions[1](
                                                 tf - tb2)[1] - a[2] * (ts - (tf - tb2)),
                                             - a[2] + 0 * ts))
                    ]

                # Initialize state piece for current 3D segment
                if dim == 0:
                    trajectory_segment = np.zeros((3, 3, ts.shape[0]))

                # Generate sampled trajectory and append to sample list
                conditions = np.array(conditions, dtype=bool)
                n = len(conditions)

                # Compute piecewise function
                samples = np.zeros((3, ts.shape[0]))
                idx = 0
                for k in range(n):
                    f = functions[k]
                    vals = ts[conditions[k]]
                    if vals.size > 0:
                        f_piece = f(vals)
                        f_piece_len = f_piece.shape[1]
                        samples[:, idx:idx + f_piece_len] = \
                            f_piece[:, :samples.shape[1] - idx]
                        idx += f_piece_len

                # Correct angular quantities from yaml format (multiply by pi)
                if self.coordinate_type == 'cylindrical':
                    if dim == 1:
                        samples *= np.pi
                elif self.coordinate_type == 'spherical':
                    if dim in (1, 2):
                        samples *= np.pi

                # Insert segment samples into piecewise trajectory
                trajectory_segment[dim] = samples

                # Calculate debug info
                logging.debug('tf: {:+1.3} (s)'.format(tf))
                logging.debug('final_vel: {:+1.3} (m/s)'.
                              format((samples[0][-1] - samples[0][-2])
                                     / self.sample_chunk_period))
                logging.debug('final_pos: {:+1.3} (m)'.format(samples[0][-1]))

                # Correct waypoints before continuing
                if reverse:
                    for idx in range(len(a) - 1):
                        a[idx], b[idx] = -a[idx], -b[idx]

            # Append 3D state pice of piecewise function to trajectory samples
            # Creates one continuous array of 3x3 state matrices
            if wp_idx == 0:
                self.trajectory_samples = \
                    np.dstack((self.trajectory_samples, trajectory_segment))
            else:
                self.trajectory_samples = \
                    np.dstack((self.trajectory_samples,
                               trajectory_segment[:, :, 1:]))

            # Incriment waypoints
            wp1 = wp2

    def trajectory_to_csv(self):
        fp = open('trajectory.csv', 'w')

        time = np.array(list(range(self.trajectory_samples[0][0].shape[0]))) / self.sample_rate
        for t_idx, t in enumerate(time):
            fp.write(str(t) + ',')
            for i in range(3):
                for j in range(3):
                    val = (self.trajectory_samples[j][i][t_idx])
                    fp.write(str(val) + ',')
            fp.write('\n')
        fp.close()

    def plot_trajectory(self, publish=True):
        """Plot 3D trajectory on a 2D graph."""
        #

        self.trajectory_to_csv()
        #
        # print('exiting in plot_trajectory for test purposes...')
        # exit()

        if publish:
            import matplotlib

            matplotlib.rcParams['font.sans-serif'] = "Latan Modern Roman"
            matplotlib.rcParams['font.family'] = "serif"
            matplotlib.rcParams['font.size'] = 8
            matplotlib.rcParams['text.usetex'] = True

            # In inches
            fig_size_x = 3.4875;
            fig_size_y = 2.5;

            from cycler import cycler
            monochrome = (cycler('color', ['k']) * cycler('marker', ['', '.']) *
                          cycler('linestyle', ['-', '--', ':', '=.']))
            plt.rc('axes', prop_cycle=monochrome)
        else:
            fig.suptitle('Target Position vs. Time', fontsize=10)

        fig = plt.figure(figsize=(fig_size_x, fig_size_y));

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)

        plt.subplots_adjust(left=0.185, bottom=0.15, right=0.999, top=0.98, wspace=0.2, hspace=0.2)

        def m2rad(x):
            return x/np.pi


        def rad2m(x):
            return np.pi * x

        for i in range(2):
            ax = plt.subplot(211 + i)
            axis_type = self.coordinate_types[self.coordinate_type]
            for j in [1]:
                plt.plot((np.array(list(range(
                    self.trajectory_samples[j][i].shape[0]))) / self.sample_rate),
                    (self.trajectory_samples[j][i])/np.pi, label=axis_type[j] + '-axis')

            import matplotlib.ticker as tck
            ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%3.2f $\pi$'))

            if i == 0:
                # plt.legend()
                plt.ylabel('Angle (rad)', fontsize=8)
                plt.xticks([])

                ax.yaxis.set_major_locator(tck.MultipleLocator(base=0.25))
            if i == 1:
                plt.ylabel('Velocity (rad$\cdot$s$^{-1}$)', fontsize=8)

                ax.yaxis.set_major_locator(tck.MultipleLocator(base=0.5))
            if i == 2:
                plt.ylabel('Acceleration (m/s/s)', fontsize=8)
        plt.xlabel('Time (s)')

        if publish:
            plt.savefig('trajectory.pdf', format='pdf')
        else:
            plt.show()
            plt.close()

    def generate_array_samples(self):
        """
        Compute radar measurements at each trajectory sample location.

        Cycle through radar array and gnerate synthetic data modeling radar
        type, noise, and other factors.

        Algorithm:
            1. Compute velocity normal to radar for Doppler calculation
            2. Compute time domain Doppler frequency for each sampled location
                x[n] = [cos(w[n] * n * T), sin(w[n] * n * T)]
        """
        # NOTE: copying data into TimeSeries ONLY, not buffer for processing.
        # Compute number of full samples in array
        num_sample_chunks = self.trajectory_samples.shape[2] // self.sample_chunk_size
        logging.debug('num_samples: {:}'.format(num_sample_chunks))

        # Allocate doppler sample array
        doppler_samples_size = (self.num_channels // 2, 2,
                                self.trajectory_samples.shape[-1])
        doppler_samples = np.empty(doppler_samples_size, dtype=np.float64)

        # Generate doppler response from trajectory for each radar in array
        for radar_idx, radar in enumerate(self.array['receivers']):
            logging.debug("generate_array_samples (%-completion): {:6.2f}".
                          format(radar_idx * 100.0 / len(self.array)))
            doppler_samples[radar_idx] = \
                self.generate_carrier_waveform(radar, self.trajectory_samples)

            # complex_data = doppler_samples[radar_idx][0] + doppler_samples[radar_idx][1] * 1.0j
            # fft_velocity = self.compute_cfft_velocity(complex_data, fft_window=2)
            # plt.plot(fft_velocity, label='fft_velocity')
            # plt.ylabel('frequency')
            # plt.show()

        # Allocate radar sample array (I & Q)
        radar_samples_size = (self.num_channels, doppler_samples.shape[-1])
        radar_samples = np.empty(radar_samples_size, dtype=np.float64)

        # TODO: use columns specified in yaml file
        # radar_samples[0, :] = doppler_samples[0][0]
        # radar_samples[1, :] = doppler_samples[0][1]
        # radar_samples[2, :] = doppler_samples[1][0]
        # radar_samples[3, :] = doppler_samples[1][1]
        # radar_samples[4, :] = doppler_samples[2][0]
        # radar_samples[5, :] = doppler_samples[2][1]

        radar_samples[0, :] = doppler_samples[0][0]
        radar_samples[1, :] = doppler_samples[0][1]
        radar_samples[2, :] = doppler_samples[1][0]
        radar_samples[3, :] = doppler_samples[1][1]

        # if 'i-channel' in self.array[0]:
        #     for idx, radar in enumerate(self.array):
        #         i_ch = radar['i-channel'] - 1
        #         q_ch = radar['q-channel'] - 1
        #         radar_samples[i_ch] = doppler_samples[idx][0]
        #         radar_samples[q_ch] = doppler_samples[idx][1]
        # else:
        #     for rad_idx in range(len(self.array)):
        #         ch_idx = rad_idx * 2
        #         radar_samples[ch_idx:ch_idx + 2, :] = doppler_samples[rad_idx]

        #
        # plt.subplot(211)
        # complex_data = radar_samples[0] + radar_samples[1] * 1.0j
        # fft_velocity = self.compute_cfft_velocity(complex_data, fft_window=2048, overlap=0)
        # plt.plot(fft_velocity, label='fft_velocity')
        # plt.ylabel('frequency')
        #
        # plt.subplot(212)
        # complex_data = radar_samples[2] + radar_samples[3] * 1.0j
        # fft_velocity = self.compute_cfft_velocity(complex_data, fft_window=2048, overlap=0)
        # plt.plot(fft_velocity, label='fft_velocity')
        # plt.ylabel('frequency')
        # plt.show()

        # Add sample chunks to SynthDAQ buffer
        for sample_chunk_idx in range(num_sample_chunks):
            # Get sample chunk slice
            start_idx = sample_chunk_idx * self.sample_chunk_size
            end_idx = start_idx + self.sample_chunk_size
            array_chunk = radar_samples[..., start_idx:end_idx]
            self.ts_buffer.append(array_chunk)

    def parallel_waveform_gen(self, idx, trajectory_sample, radar):
        print('Entered parallel_waveform_gen...')
        # Loop over all 3x3 StateMatrix's in the sample chunk

        # Get constants for conversion
        c = spc.speed_of_light

        waveform = radar['waveform']
        antenna = radar['antenna']

        # Gather waveform parameters
        f0 = waveform['fc']
        prf = waveform['prf']
        pw = waveform['pw']
        bw = waveform['bw']
        type = waveform['type']

        # Gather antenna parameters
        antenna_cos_exp = antenna['cosine_exponent']

        sample = StateMatrix(trajectory_sample,
                             coordinate_type=self.coordinate_type)
        # TODO: this ^ should really be the position at t-tau, not t.

        # Get sample at radar location
        sample_rad = sample.get_state(coordinate_type='spherical',
                                      origin=Point(*radar['location']))

        # Compute phase/amplitude of the signal
        rho = sample_rad[0, 0]
        theta = sample_rad[2, 0]
        # Time delay of signal
        tau = 2 * (rho / c)
        sample_interval = self.sample_interval
        time = sample_interval * idx

        def T(t, pw):
            """Compute the rollover of the LFM chirp."""
            return t % pw

        # If the transmitter was on @ current_time - propagation_time
        if (time - tau) % (1.0 / prf) < pw:
            if type == 'lfm':
                phase_d = 2 * np.pi * f0 * tau
                phase_lfm = -np.pi * (bw / pw) * \
                    (T(time, pw)**2 - T(time - tau, pw)**2)
                complex_signal = np.exp(1.0j * phase_d + 1.0j * phase_lfm)
                # print('s({})={}'.format(time, complex_signal))
                output_i = complex_signal.real
                output_q = complex_signal.imag
            elif type == 'fmcw':
                pwc = pw / 2
                # Up-chrip
                if time - tau % pw < pwc:
                    phase = np.pi * (bw / pwc) * (time**2 + (time - tau)**2)
                # Down-chirp
                else:
                    phase = -np.pi * (bw / pwc)\
                        * (2 * pw * (time - tau) + (time - tau)**2 - 2 * tau * time
                           + time**2)
                output_i = np.cos(phase)
                output_q = np.sin(phase)
            else:
                phase = 2 * np.pi * f0 * (time - tau)
                output_i = np.cos(phase)\
                    # + np.random.normal(0, 1.0)
                output_q = np.sin(phase)\
                    # + np.random.normal(0, 1.0)

            iq_sample = np.array((output_i, output_q))\
                * np.cos(theta)**antenna_cos_exp\
                # * theta**(1/5)
        else:
            iq_sample = np.array((0, 0))

        return iq_sample

    def splat_parallel_waveform_gen(self, args):
        """Splats args into function f."""
        print(args)
        return self.parallel_waveform_gen(*args)

    def generate_carrier_waveform(self, rx, trajectory_sample):
        """Create new carrier waveform samples.

        Args:
            radar (dictionary): Dictionary object describing radar
                location, waveform, and antenna parameters, etc. based on YAML.

            trajectory_sample (numpy.ndarray(3, 3, sample_chunk_size)): Sample
                of trajectory data the length of one DAQ sample.

        Returns:
            numpy.ndarray(num_channels, sample_chunk_size): Array of sampled
                data. num_channels may be 1 or 2 depending on if radar is
                complex.

        """
        # Allocate space for samples
        doppler_sample_chunk = np.empty((trajectory_sample.shape[-1], 2))

        # Get constants for conversion
        c = spc.speed_of_light

        # Iterate over all transmitter receiver pairs
        rx_cos_exp = self.hpbw_to_cos(rx['antenna']['hpbw'])

        for tx in self.array['transmitters']:

            waveform = tx['waveform']
            tx_hpbw = tx['antenna']['hpbw']

            # Gather waveform parameters
            f0 = waveform['f0']
            prf = waveform['prf']
            bw = waveform['bw']

            # If no PW specified, set to 1/PRF – continuous-wave
            if 'pw' in waveform.keys():
                pw = waveform['pw']
            else:
                pw = 1/prf

            waveform_type = waveform['type']
            sample_interval = self.sample_interval

            # Gather antenna parameters
            tx_cos_exp = self.hpbw_to_cos(tx['antenna']['hpbw'])

            # Loop over all 3x3 StateMatrix's in the sample chunk
            for idx in range(trajectory_sample.shape[-1]):

                # Print percent-competion message
                if idx % 10000 == 0:
                    percent_complete = idx / trajectory_sample.shape[-1] * 100
                    logging.debug("radar_samples (%-completion): {:6.2f}".
                                  format(percent_complete))

                sample = StateMatrix(trajectory_sample[..., idx],
                                     coordinate_type=self.coordinate_type)
                # TODO: this ^ should really be the position at t-tau, not t.

                # Get sample at radar location
                rx_rad = sample.get_state(coordinate_type='spherical',
                                          origin=Point(*rx['location']))
                tx_rad = sample.get_state(coordinate_type='spherical',
                                          origin=Point(*tx['location']))

                # Compute phase/amplitude of the signal
                rx_rho = rx_rad[0, 0]
                rx_theta = rx_rad[2, 0]
                tx_rho = tx_rad[0, 0]
                tx_theta = tx_rad[2, 0]

                # Time delay of signal
                tau = (rx_rho + tx_rho) / spc.c
                time = sample_interval * idx

                def T(t, pw):
                    """Compute the rollover of periodic FM/PM signals."""
                    return np.mod(t, pw)

                # If the transmitter was on @ current_time - propagation_time
                # Handles pulsed Doppler
                if (time - tau) % (1.0 / prf) < pw:

                    # Calculate carrier phases
                    carrier_tx_phase = 2.0 * np.pi * f0 * time
                    carrier_rx_phase = 2.0 * np.pi * f0 * (time - tau)

                    # TODO convert phase for lfm and fmcw to functions
                    if waveform_type == 'lfm':
                        lfm_tx_phase = np.pi * (bw / pw) * T(time,pw)**2
                        lfm_rx_phase = np.pi * (bw / pw) * T(time - tau,pw)**2

                        total_tx_phase = carrier_tx_phase + lfm_tx_phase
                        total_rx_phase = carrier_rx_phase + lfm_rx_phase
                    elif waveform_type == 'fmcw':
                        # Alternate up and down chirp every pw seconds
                        # Compute transmitted chrip
                        if (time // pw) % 2 == 0:
                            # up-chirp
                            lfm_tx_phase = np.pi * (bw / pw) * T(time,pw)**2
                        else:
                            # down-chirp
                            lfm_tx_phase = np.pi * (bw / pw) * (2 * pw * T(time,pw) - T(time,pw)**2)

                        # Compute received chrip
                        td = (time - tau)
                        if (td // pw) % 2 == 0:
                            # up-chirp
                            lfm_rx_phase = np.pi * (bw / pw) * T(td,pw)**2
                        else:
                            # down-chirp
                            lfm_rx_phase = np.pi * (bw / pw) * (2 * pw * T(td,pw) - T(td,pw)**2)

                        total_tx_phase = carrier_tx_phase + lfm_tx_phase
                        total_rx_phase = carrier_rx_phase + lfm_rx_phase
                    elif waveform_type == 'cw':
                        total_tx_phase = carrier_tx_phase
                        total_rx_phase = carrier_rx_phase
                    else:
                        raise TypeError('Wavform type not supported')

                    carrier_tx = np.exp(1.0j * total_tx_phase)
                    carrier_rx = np.exp(1.0j * total_rx_phase)

                    # TODO: Add fading effect
                    complex_if = carrier_rx * np.conjugate(carrier_tx)

                    output_i = complex_if.real
                    output_q = complex_if.imag

                    # Apply cosine beam approximation
                    iq_sample = np.array((output_i, output_q)) \
                                * np.cos(rx_theta)**rx_cos_exp \
                                * np.cos(tx_theta)**tx_cos_exp
                else:
                    iq_sample = np.array((0, 0))

                doppler_sample_chunk[idx, :] = iq_sample

        return doppler_sample_chunk.transpose()

    def get_samples(self):
        """
        Control generation of DAQ data.

        Required function called by sample_loop method of pyratk.daq.DAQ base
        class to copy data into the buffers.
        """
        # TODO: Remove real-time availability
        #       use VirtualDAQ for playback
        # TODO: Update base class to _get_samples(self)

        # self._generate_new_sample(self.sample_num)
        #
        # self.buffer.append((self.data, self.sample_num))
        # self.ts_buffer.append(self.data)
        #
        # self.sample_num += 1
        # # Set the update event to True once data is read in
        # self.data_available.set()
        #
        # # If running in "real-time," add delay to thread
        # if not self.time_warp:
        #     sample_chunk_period = self.sample_chunk_size / self.sample_rate
        #     time.sleep(sample_chunk_period)
        pass

    def reset(self):
        self.buffer = []
        self.ts_buffer.clear()
        self.sample_num = 0
        self.reset_flag = True
        self.coordinate_type = None
        self.waypoints = []
        self.trajectory_samples = np.zeros((3, 3, 0))
        self.radar_samples = []

    # === DEBUGGING FUNCTIONS ===
    def compute_cfft_velocity(self, complex_data, overlap=0, fft_window=2048,
                              fft_size=2**18, f0=24.15e9):
        # Get constants for conversion
        # c = spc.speed_of_light

        velocity_list = []

        for idx in range(complex_data.shape[-1] // fft_window):
            start_idx = idx * fft_window - overlap
            if start_idx < 0:
                start_idx = 0
            end_idx = start_idx + fft_window
            complex_data_window = complex_data[start_idx:end_idx]

            fft_complex = np.fft.fft(complex_data_window, fft_size)
            fft_complex = np.fft.fftshift(fft_complex)

            fft_mag = np.linalg.norm([fft_complex.real, fft_complex.imag],
                                     axis=0)

            bin_size = self.sample_rate / fft_mag.shape[0]
            bin = np.argmax(fft_mag).astype(np.int32)
            freq_fft = (bin - fft_mag.shape[0] / 2) * bin_size

            # wavelength = c / f0
            # fft_velocity = freq_fft * wavelength / 2
            # velocity_list.append(fft_velocity)
            velocity_list.append(freq_fft)

        return velocity_list

    def hpbw_to_cos(self, hpbw):
        return np.log10(0.5) / np.log10(np.cos(hpbw / 2.0))
