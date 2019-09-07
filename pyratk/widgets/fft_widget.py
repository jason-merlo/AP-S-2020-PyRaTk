# -*- coding: utf-8 -*-
"""
FFT Widget Class.

Contains RadarWidget class used to draw the max frequency and FFT graphs

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import pyqtgraph as pg          # Used for RadarWidget superclass
import numpy as np              # Used for numerical operations
import time                     # Used for FPS calculations


class FftWidget(pg.GraphicsLayoutWidget):
    def __init__(self, radar, vmax_len=100, show_max_plot=False):
        super(FftWidget, self).__init__()

        # Copy arguments to member variables
        self.data_mgr = radar.data_mgr
        self.source = self.data_mgr.source
        self.radar = radar
        self.vmax_len = vmax_len
        self.update_period = \
            self.source.sample_chunk_size / self.source.sample_rate
        self.show_max_plot = show_max_plot

        # Add FFT max plot to layout
        if show_max_plot:
            self.vmax_plot = self.addPlot()

            # Set up fmax plot
            self.vmax_plot.setDownsampling(mode='peak')
            self.vmax_plot.setClipToView(True)
            self.vmax_plot.setRange(xRange=[-vmax_len, 0], yRange=[-0.5, 0.5])
            self.vmax_plot.setLimits(
                xMax=0, yMax=20, yMin=-20)
            self.vmax_pw = self.vmax_plot.plot()

            self.vmax_line = pg.InfiniteLine(angle=0, movable=False)
            self.vmax_plot.addItem(self.vmax_line)

            # self.a_pw = self.vmax_plot.plot()
            self.vmax_plot.setLabel('left', text="Radial Velocity", units="m/s")
            self.vmax_ax_bottom = self.vmax_plot.getAxis('bottom')
            self.vmax_ax_bottom.setScale(self.update_period)
            self.vmax_plot.setLabel('bottom', text="Time", units="s")

            # Create next row for FFT plot
            self.nextRow()

        # Calculate reasonable ranges for FFT peak outputs
        fft_xrange = [-50 / self.radar.bin_size, 50 / self.radar.bin_size]
        fft_yrange = [-100, 0]

        # Add FFT plot
        self.fft_plot = self.addPlot()

        # Set up fft plot
        self.fft_plot.setDownsampling(mode='peak')
        self.fft_plot.setClipToView(True)
        # self.fft_plot.setLogMode(x=False, y=True)  # Log Y-axis of FFT views
        self.fft_plot.setRange(disableAutoRange=True,
                               xRange=fft_xrange, yRange=fft_yrange)
        self.fft_plot.setLimits(
            xMin=fft_xrange[0], xMax=fft_xrange[1], yMin=-80, yMax=0)
        self.fft_pw = self.fft_plot.plot()
        self.fft_max_freq_line = pg.InfiniteLine(angle=90, movable=False)
        self.fft_max_pwr_line = pg.InfiniteLine(angle=0, movable=False)
        self.fft_plot.addItem(self.fft_max_freq_line)
        self.fft_plot.addItem(self.fft_max_pwr_line)
        self.fft_ax_bottom = self.fft_plot.getAxis('bottom')
        self.fft_ax_bottom.setScale(self.radar.bin_size)
        self.fft_plot.setLabel('bottom', text="Frequency", units="Hz")
        self.fft_plot.setLabel('left', text="Power", units="dBm")

        # FPS ticker data
        self.lastTime = time.time()
        self.fps = None

    def update_vmax(self):
        # Update fmax graph
        vmax_data = self.radar.ts_drho.data
        vmax_ptr = self.radar.ts_drho.head_ptr
        self.vmax_pw.setData(vmax_data, pen=pg.mkPen({'color': "FFF"}))
        self.vmax_pw.setPos(-vmax_ptr, 0)

        self.vmax_line.setValue(vmax_data[-1])

        # a_data = self.radar.ts_a.data
        # a_ptr = self.radar.ts_a.head_ptr
        # self.a_pw.setData(a_data, pen=pg.mkPen({'color': "FF0"}))
        # self.a_pw.setPos(-a_ptr, 0)

        try:
            self.vmax_plot.setTitle(
                'Max Velocity:\t{:+0.4f} (m/s)'.format(vmax_data[-1]))
        except Exception:
            pass

    def update_fft(self):
        if self.radar.cfft_data is not None:
            log_fft = 10 * np.log(self.radar.cfft_data)
            max_log_fft = np.max(log_fft)
            self.fft_pw.setData(log_fft)
            self.fft_pw.setPos(-self.radar.center_bin, 0)

            # draw max FFT lines
            self.fft_max_freq_line.setValue(self.radar.fmax
                                            / self.radar.bin_size)
            self.fft_max_pwr_line.setValue(max_log_fft)

        self.fft_plot.setTitle(
            'Max Frequency:\t{:+0.4f} (Hz) @ {:+0.3f} (dBm)'.format(
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
        # self.vmax_plot.setTitle('%0.2f fps' % self.fps)
        # print('%0.2f fps' % self.fps)

    def update(self):
        self.update_fft()
        if self.show_max_plot:
            self.update_vmax()
        # self.update_fps()

    def reset(self):
        self.radar.reset()
        self.vmax_data = []
        # When paused, redraw after reset
        if self.data_mgr.paused:
            self.update()
