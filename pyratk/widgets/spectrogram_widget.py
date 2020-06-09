# -*- coding: utf-8 -*-
"""
FFT Widget Class.

Contains RadarWidget class used to draw the max frequency and FFT graphs

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import pyqtgraph as pg          # Used for RadarWidget superclass
import numpy as np              # Used for numerical operations
from scipy import signal        # Used for upsampling
import time                     # Used for FPS calculations
from matplotlib import cm       # Used for colormaps


class SpectrogramWidget(pg.PlotWidget):
    def __init__(self, radar, spectrogram_length=100, show_max_plot=False,
                 fft_yrange=[-90,20], fft_xrange=[-5e3,5e3]):
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
        self.speed = 10
        self.downsample = 10

        # Add FFT max plot to layout
        # if show_max_plot:
        #     self.fmax_data = []
        #
        #     self.fmax_plot = self.addPlot()
        #
        #     # Set up fmax plot
        #     self.fmax_plot.setDownsampling(mode='peak')
        #     self.fmax_plot.setClipToView(True)
        #     # self.fmax_plot.setRange(xRange=[-fmax_len, 0], yRange=[-0.5, 0.5])
        #     self.fmax_plot.setRange(xRange=[-fmax_len, 0])
        #     # self.fmax_plot.setLimits(
        #     #     xMax=0, yMax=20, yMin=-20)
        #     self.fmax_pw = self.fmax_plot.plot()
        #
        #     # self.fmax_line = pg.InfiniteLine(angle=0, movable=False)
        #     # self.fmax_plot.addItem(self.fmax_line)
        #
        #     # self.a_pw = self.fmax_plot.plot()
        #     self.fmax_plot.setLabel('left', text="Frequency", units="Hz")
        #     self.fmax_ax_bottom = self.fmax_plot.getAxis('bottom')
        #     self.fmax_ax_bottom.setScale(self.update_period)
        #     self.fmax_plot.setLabel('bottom', text="Chirps", units="")
        #
        #     # Create next row for FFT plot
        #     self.nextRow()
        #
        # # Calculate reasonable ranges for FFT peak outputs
        # # fft_xrange = [-50 / self.radar.bin_size, 50 / self.radar.bin_size]
        # # fft_yrange = [-100, 0]
        #
        # # Add FFT plot
        # self.fft_plot = self.addPlot()
        #
        # # Set up fft plot
        # self.fft_plot.setDownsampling(mode='peak')
        # self.fft_plot.setClipToView(True)
        # # self.fft_plot.setLogMode(x=False, y=True)  # Log Y-axis of FFT views
        # self.fft_plot.setRange(disableAutoRange=True,
        #                        xRange=np.array(fft_xrange)/self.radar.bin_size,
        #                        yRange=fft_yrange)
        # self.fft_plot.setLimits(
        #     xMin=-self.source.sample_rate/(2*self.radar.bin_size),
        #     xMax=self.source.sample_rate/(2*self.radar.bin_size),
        #     yMin=fft_yrange[0], yMax=fft_yrange[1])
        # self.fft_pw = self.fft_plot.plot()
        # self.fft_max_freq_line = pg.InfiniteLine(angle=90, movable=False)
        # self.fft_max_pwr_line = pg.InfiniteLine(angle=0, movable=False)
        # self.fft_plot.addItem(self.fft_max_freq_line)
        # self.fft_plot.addItem(self.fft_max_pwr_line)
        # self.fft_ax_bottom = self.fft_plot.getAxis('bottom')
        # self.fft_ax_bottom.setScale(self.radar.bin_size)
        # self.fft_plot.setLabel('bottom', text="Frequency", units="Hz")
        # self.fft_plot.setLabel('left', text="Magnitude", units="dBV")

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

        # setup the correct scaling for y-axis
        # log_freq_range = np.linspace(0, np.log10(self.source.chunk_size),
        #                              self.radar.cfft.size)
        # yscale = (log_freq_range[-1] / self.img_array.shape[0])
        # self.img.scale((1. / self.source.sample_rate) * self.source.chunk_size / self.speed, yscale)
        # self.setLogMode(y=True)
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

        # self.setLabel('right', 'Range', 'm')
        # right_axis=self.getAxis('right')
        # right_axis.setScale((log_freq_range[-1] / self.img_array.shape[0]) * (FC / (2*spc.c)))


    def update_spectrogram(self):
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
