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
    def __init__(self, receiver, xrange=[-50,50], yrange=[-15e3,0]):
        super().__init__()

        # Copy arguments to member variables
        self.daq = receiver.daq
        self.source = self.daq.source
        self.receiver = receiver
        self.update_period = \
            self.source.sample_chunk_size / self.source.sample_rate

        self.xrange = xrange
        self.yrange = yrange

        # TODO temp
        self.downsample = 1

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
        self.img.setLevels([-80, 10]) # Good for drone
        #self.img.setLevels([-10, 20]) # Good for drone

        self.rescale()

        self.setLabel('left', 'Frequency', units='Hz')
        self.setLabel('bottom', 'Frequency', units='Hz')
        self.showGrid(x=True, y=True)

        left_axis=self.getAxis('left')
        left_axis.setGrid(255)

        # Invert y-axis so negative is "up" (corresponds with range)
        self.getViewBox().invertY(True)

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

        self.img.scale(self.receiver.slow_bin_size * self.downsample,
                       self.receiver.fast_bin_size * self.downsample)

        self.setRange(disableAutoRange=True, yRange=np.array(self.yrange))

        self.img.translate(-np.array(self.receiver.fft_mat.shape[1]) / (2 * self.downsample),
            -np.array(self.receiver.fft_mat.shape[0]) / (2 * self.downsample))


        # TODO: why is the /4 instead of /2??
        slow_limit = self.receiver.slow_bin_size * self.receiver.slow_fft_size / 4

        self.setLimits(
            xMin=-slow_limit, xMax=slow_limit)
