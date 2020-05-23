# -*- coding: utf-8 -*-
"""
IQData Widget Class.

Draw value vs time data.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import pyqtgraph as pg          # Used for RadarWidget superclass
import numpy as np              # Used for numerical operations
import time                     # Used for FPS calculations


class IQWidget(pg.GraphicsLayoutWidget):
    def __init__(self, radar, yrange=None, max_len=None):
        super(IQWidget, self).__init__()

        # Copy arguments to member variables
        self.data_mgr = radar.data_mgr
        self.source = self.data_mgr.source
        self.radar = radar
        self.update_period = \
            self.source.sample_chunk_size / self.source.sample_rate

        # Add FFT plot
        self.iq_plot = self.addPlot()

        # Set up fft plot
        self.iq_plot.setDownsampling(mode='peak')
        self.iq_plot.setClipToView(True)
        # self.iq_plot.setLogMode(x=False, y=True)  # Log Y-axis of FFT views
        # self.iq_plot.setRange(disableAutoRange=True,
        #                        xRange=fft_xrange, yRange=fft_yrange)
        # self.iq_plot.setLimits(
        #     xMin=fft_xrange[0], xMax=fft_xrange[1],
        #     yMin=fft_yrange[0], yMax=fft_yrange[1])
        self.i_pw = self.iq_plot.plot()
        self.q_pw = self.iq_plot.plot()
        self.iq_ax_bottom = self.iq_plot.getAxis('bottom')
        # self.iq_ax_bottom.setScale(self.radar.bin_size)
        self.iq_plot.setLabel('bottom', text="Time", units="S")
        self.iq_plot.setLabel('left', text="Amplitude", units="V")

        # FPS ticker data
        self.lastTime = time.time()
        self.fps = None

    def update_plot(self):
        if self.radar.ts_data.data.shape[0] != 0:
            iq_data = np.abs(self.radar.ts_data[-1])
            self.i_pw.setData(iq_data[0, :], pen=pg.mkPen({'color': "F00"}))
            self.q_pw.setData(iq_data[1, :], pen=pg.mkPen({'color': "0F0"}))

    def update_fps(self):
        now = time.time()
        dt = now - self.lastTime
        self.lastTime = now
        if self.fps is None:
            self.fps = 1.0 / dt
        else:
            s = np.clip(dt * 3., 0, 1)
            self.fps = self.fps * (1 - s) + (1.0 / dt) * s
        # print('%0.2f fps' % self.fps)

    def update(self):
        self.update_plot()
        # self.update_fps()

    def reset(self):
        # When paused, redraw after reset
        if self.data_mgr.paused:
            self.update()
