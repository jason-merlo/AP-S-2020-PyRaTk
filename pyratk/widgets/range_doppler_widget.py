# -*- coding: utf-8 -*-
"""
Data-Cube Widget Class.

Contains RadarWidget class used to visualize a radar Data-Cube represenataion
of fast- and slow-time data.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import pyqtgraph as pg          # Used for RadarWidget superclass
import numpy as np              # Used for numerical operations
from scipy import signal        # Used for upsampling
import time                     # Used for FPS calculations
from matplotlib import cm       # Used for colormaps


class DataCubeWidget(pg.PlotWidget):
    def __init__(self, radar, fast_time_yrange=[-90,20],
        fast_time_xrange=[-5e3,5e3]):
        super(SpectrogramWidget, self).__init__()

        # Copy arguments to member variables
        self.data_mgr = radar.data_mgr
        self.source = self.data_mgr.source
        self.radar = radar
        self.spectrogram_length = spectrogram_length
        self.update_period = \
            self.source.sample_chunk_size / self.source.sample_rate
        self.show_max_plot = show_max_plot

        # TODO temp
        self.downsample = 10

        # FPS ticker data
        self.lastTime = time.time()
        self.fps = None

        # -----

        self.img = pg.ImageItem()
        self.addItem(self.img)

        # Allocate image array to store spectrogram
        self.img_array = np.zeros((int(self.radar.cfft_data.size / self.downsample), spectrogram_length))

        # Get the colormap
        colormap = cm.get_cmap("jet")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-90, 0])

        self.img.scale(1, self.radar.bin_size * self.downsample)

        self.setRange(disableAutoRange=True, yRange=np.array(fft_xrange))
        self.setLimits(
            yMin=-self.source.sample_rate/2, yMax=self.source.sample_rate/2)

        self.img.translate(0, -self.radar.cfft_data.size / (2 * self.downsample))

        self.setLabel('left', 'Frequency', units='Hz')
        self.showGrid(x=False, y=True)

        left_axis=self.getAxis('left')
        left_axis.setGrid(255)

        self.img.setCompositionMode(pg.QtGui.QPainter.CompositionMode_Plus)


    def update_datacube(self):
        if self.radar.cfft_data is not None:
            pass

    def update_plot(self):
        ALPHA = 0.75
        AVG_SAMPLES = 2*self.speed

        if self.radar.cfft_data is not None:

            downsampled = self.radar.cfft_data[::self.downsample]

            log_fft = 10 * np.log(downsampled)

            avg_samples = self.img_array[:, -AVG_SAMPLES:]
            avg_psd = np.average(avg_samples, axis=1)
            new_psd = avg_psd * (1-ALPHA)/self.speed + log_fft * ALPHA
            self.img_array = np.roll(self.img_array, -self.speed, 1)
            self.img_array[:,-self.speed:] = np.repeat(np.expand_dims(new_psd, axis=1), self.speed,axis=1)

            self.img.setImage(self.img_array.T, autoLevels=False, autoDownsample=True)

            print(self.getAxis("left").range)

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
        self.update_spectrogram()
        self.update_fps()

    def reset(self):
        self.fmax_data = []
        # When paused, redraw after reset
        if self.data_mgr.paused:
            self.update()
