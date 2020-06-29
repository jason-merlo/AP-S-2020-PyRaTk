# -*- coding: utf-8 -*-
"""
RangeDoppler Widget Class.

Contains RadarWidget class used to range-Doppler graphs

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import pyqtgraph as pg          # Used for RadarWidget superclass
import numpy as np              # Used for numerical operations
from scipy import signal        # Used for upsampling
import time                     # Used for FPS calculations
from matplotlib import cm       # Used for colormaps


class RangeDopplerWidget(pg.PlotWidget):
    def __init__(self, receiver, xrange=[-50,50], yrange=[-50e3,0], showMeters=True):
        super().__init__()

        # Copy arguments to member variables
        self.daq = receiver.daq
        self.source = self.daq.source
        self.receiver = receiver
        self.pulse = self.receiver.transmitter.pulses[0]
        self.showMeters = showMeters
        self.update_period = \
            self.source.sample_chunk_size / self.source.sample_rate

        self.freq_to_vel = (3e8 / self.pulse.fc) / 2
        self.freq_to_range = -(self.pulse.delay / self.pulse.bw) * 3e8 / 2

        self.offset_freq = 5.5 / self.freq_to_range * 2

        self.xrange = xrange
        self.yrange = yrange

        # TODO temp
        self.downsample = 1 # MUST EQUAL 1

        # FPS ticker data
        self.lastTime = time.time()
        self.fps = None

        # -----

        self.img = pg.ImageItem()
        self.addItem(self.img)

        # Allocate image array to store spectrogram
        # self.img_array = np.zeros(np.around(np.array(self.receiver.fft_mat.shape) / self.downsample).astype(np.int))

        # Get the colormap
        colormap = cm.get_cmap("jet")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-90, -10]) # Good for drone
        #self.img.setLevels([-10, 20]) # Good for drone

        self.rescale()

        if showMeters:
            self.setLabel('left', 'Range', units='m')
            self.setLabel('bottom', 'Velocity', units='m/s')
        else:
            self.setLabel('left', 'Frequency', units='Hz')
            self.setLabel('bottom', 'Frequency', units='Hz')
            # Invert y-axis so negative is "up" (corresponds with range)
            self.getViewBox().invertY(True)
        self.showGrid(x=True, y=True)

        left_axis=self.getAxis('left')
        left_axis.setGrid(255)

        self.img.setCompositionMode(pg.QtGui.QPainter.CompositionMode_Plus)

        # self.setLabel('right', 'Range', 'm')
        # right_axis=self.getAxis('right')
        # right_axis.setScale((log_freq_range[-1] / self.img_array.shape[0]) * (FC / (2*spc.c)))


    def update_map(self):

        if self.receiver.fft_mat is not None:

            downsampled = self.receiver.fft_mat[::self.downsample, ::self.downsample]
            # print('rd - downsampled.shape', downsampled.shape)

            try:
                if not np.all(self.receiver.fast_fft_data == 0):
                    log_fft = 10 * np.log(downsampled)
                    # print('log_fft.shape', log_fft.shape)
                    self.img.setImage(log_fft, autoLevels=False, autoDownsample=True)
            except:
                pass

            # print(self.getAxis("left").range)

    def update_fps(self):
        now = time.time()
        dt = now - self.lastTime
        self.lastTime = now
        if self.fps is None:
            self.fps = 1.0 / dt
        else:
            s = np.clip(dt * 3., 0, 1)
            self.fps = self.fps * (1 - s) + (1.0 / dt) * s
        print('%0.2f fps' % self.fps)

    def update(self):
        self.update_map()
        #self.update_fps()

    def reset(self):
        # When paused, redraw after reset
        if self.daq.paused:
            self.update()
        self.rescale()

    def rescale(self):
        self.img.resetTransform()

        self.update_period = \
            self.source.sample_chunk_size / self.source.sample_rate

        if self.showMeters:
            slow_scale = self.receiver.slow_bin_size * self.freq_to_vel
            fast_scale = self.receiver.fast_bin_size * self.freq_to_range
            yrange = np.array(self.yrange) * self.freq_to_range
            slow_limit = self.receiver.slow_bin_size * self.receiver.slow_fft_size / 4 * self.freq_to_vel
        else:
            slow_scale = self.receiver.slow_bin_size
            fast_scale = self.receiver.fast_bin_size
            yrange = self.yrange
            slow_limit = self.receiver.slow_bin_size * self.receiver.slow_fft_size / 4


        self.img.scale(slow_scale, fast_scale)

        self.setRange(disableAutoRange=True, yRange=np.array(yrange))

        offset_bins = self.offset_freq / self.receiver.fast_fft_size

        self.img.translate(-np.array(self.receiver.fft_mat.shape[1]) / (2 * self.downsample),
            -np.array(self.receiver.fft_mat.shape[0]) / (2 * self.downsample))

        self.setLimits(
            xMin=-slow_limit, xMax=slow_limit)
