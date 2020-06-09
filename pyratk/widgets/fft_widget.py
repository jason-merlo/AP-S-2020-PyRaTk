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


class FftWidget(pg.GraphicsLayoutWidget):
    def __init__(self, radar, fmax_len=100, show_max_plot=False,
                 fft_yrange=[-90,20], fft_xrange=[-25e3,25e3]):
        super(FftWidget, self).__init__()

        # Copy arguments to member variables
        self.daq = radar.daq
        self.source = self.daq.source
        self.radar = radar
        self.fmax_len = fmax_len
        self.update_period = \
            self.source.sample_chunk_size / self.source.sample_rate
        self.show_max_plot = show_max_plot

        # Add FFT max plot to layout
        if show_max_plot:
            self.fmax_data = []

            self.fmax_plot = self.addPlot()

            # Set up fmax plot
            self.fmax_plot.setDownsampling(mode='peak')
            self.fmax_plot.setClipToView(True)
            # self.fmax_plot.setRange(xRange=[-fmax_len, 0], yRange=[-0.5, 0.5])
            self.fmax_plot.setRange(xRange=[-fmax_len, 0])
            # self.fmax_plot.setLimits(
            #     xMax=0, yMax=20, yMin=-20)
            self.fmax_pw = self.fmax_plot.plot()

            # self.fmax_line = pg.InfiniteLine(angle=0, movable=False)
            # self.fmax_plot.addItem(self.fmax_line)

            # self.a_pw = self.fmax_plot.plot()
            self.fmax_plot.setLabel('left', text="Frequency", units="Hz")
            self.fmax_ax_bottom = self.fmax_plot.getAxis('bottom')
            self.fmax_ax_bottom.setScale(self.update_period)
            self.fmax_plot.setLabel('bottom', text="Chirps", units="")

            # Create next row for FFT plot
            self.nextRow()

        # Calculate reasonable ranges for FFT peak outputs
        # fft_xrange = [-50 / self.radar.fast_bin_size, 50 / self.radar.fast_bin_size]
        # fft_yrange = [-100, 0]

        # Add FFT plot
        self.fft_plot = self.addPlot()

        # Set up fft plot
        self.fft_plot.setDownsampling(mode='peak')
        self.fft_plot.setClipToView(True)
        # self.fft_plot.setLogMode(x=False, y=True)  # Log Y-axis of FFT views
        self.fft_plot.setRange(disableAutoRange=True,
                               xRange=np.array(fft_xrange)/self.radar.fast_bin_size,
                               yRange=fft_yrange)
        self.fft_plot.setLimits(
            xMin=-self.source.sample_rate/(2*self.radar.fast_bin_size),
            xMax=self.source.sample_rate/(2*self.radar.fast_bin_size),
            yMin=fft_yrange[0], yMax=fft_yrange[1])
        self.fft_pw = self.fft_plot.plot()
        self.fft_max_freq_line = pg.InfiniteLine(angle=90, movable=False)
        self.fft_max_pwr_line = pg.InfiniteLine(angle=0, movable=False)
        self.fft_plot.addItem(self.fft_max_freq_line)
        self.fft_plot.addItem(self.fft_max_pwr_line)
        self.fft_ax_bottom = self.fft_plot.getAxis('bottom')
        self.fft_ax_bottom.setScale(self.radar.fast_bin_size)
        self.fft_plot.setLabel('bottom', text="Frequency", units="Hz")
        self.fft_plot.setLabel('left', text="Magnitude", units="dBV")

        # FPS ticker data
        self.lastTime = time.time()
        self.fps = None

    def update_fmax(self):
        # Update fmax graph
        if not np.all(self.radar.cfft_data == self.radar.cfft_data[0]):
            self.fmax_data.append(self.radar.fmax)
        self.fmax_pw.setData(self.fmax_data, pen=pg.mkPen({'color': "FFF"}))
        self.fmax_pw.setPos(-len(self.fmax_data), 0)

        # self.fmax_line.setValue(fmax_data[-1])

        # a_data = self.radar.ts_a.data
        # a_ptr = self.radar.ts_a.head_ptr
        # self.a_pw.setData(a_data, pen=pg.mkPen({'color': "FF0"}))
        # self.a_pw.setPos(-a_ptr, 0)

        try:
            self.fmax_plot.setTitle(
                'Max Velocity:\t{:+0.4f} (m/s)'.format(fmax_data[-1]))
        except Exception:
            pass

    def update_fft(self):
        if self.radar.cfft_data is not None:
            log_fft = 10 * np.log(self.radar.cfft_data)
            self.fft_pw.setData(log_fft)
            self.fft_pw.setPos(-self.radar.center_bin, 0)

            # draw max FFT lines
            max_log_fft = np.nanmax(log_fft)
            self.fft_max_freq_line.setValue(self.radar.fmax
                                            / self.radar.fast_bin_size)
            self.fft_max_pwr_line.setValue(max_log_fft)

        self.fft_plot.setTitle(
            'Max Frequency:\t{:+0.4f} (Hz) @ {:+0.3f} (dBV)'.format(
                self.radar.fmax, max_log_fft))

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
        self.update_fft()
        if self.show_max_plot:
            self.update_fmax()
        self.update_fps()

    def reset(self):
        self.fmax_data = []
        # When paused, redraw after reset
        if self.daq.paused:
            self.update()

        self.rescale()

    def rescale(self):
        self.update_period = \
            self.source.sample_chunk_size / self.source.sample_rate

        self.fft_ax_bottom.setScale(self.radar.fast_bin_size)
