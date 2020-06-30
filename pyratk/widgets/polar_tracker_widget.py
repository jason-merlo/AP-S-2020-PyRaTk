# -*- coding: utf-8 -*-
"""
PolarTracker Widget Class.

Contains parametric graph capable of plotting a tracked object's path in the
polar coordinate system.

Author: Jason Merlo, Stavros Vakalis
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np              # Used for numerical operations
import platform                 # Get OS for DPI scaling
import math
from pyratk.datatypes.geometry import Point, Circle


class PolarTrackerWidget(pg.GraphicsLayoutWidget):
    def __init__(self, tracker, max_range=20, moving_average_weight=0.2, trail_length=150):
        super().__init__()
        """
        Initialize polar tracker widget.

        tracker - Tracker object
            Note:   Tracker requires list of namedtouples named `detections`.
                    The namedtouple must contain:
                        - location (Point): location of detection
                        - power (float): power of detection
                        - doppler (float): Doppler velocity of detection
                        - velocity (Point): Velocity vector (if tracked)
        max_range - float
            Maximum range shown on the plot

        moving_average_weight - float
            Value between 0 and 1 specifying how much weight new values have
            avg = old * (1 - weight) + new * weight
        """

        # Copy arguments to member variables
        self.tracker = tracker
        self.max_range = max_range
        self.weight = moving_average_weight

        self.det_loc = Point()

        self.trail_length = trail_length

        self.trajectory = []

        # Add plots to layout
        self.plot = self.addPlot()

        # Add polar grid lines
        self.plot.addLine(x=0, pen=0.2)
        self.plot.addLine(y=0, pen=0.2)

        for r in range(2, self.max_range*2, 2):
            circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            circle.setPen(pg.mkPen(0.2))
            self.plot.addItem(circle)

        # Add radar location marker plot
        # self.radar_loc_plot = pg.ScatterPlotItem(
        #     size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 255))

        # for radar in self.tracker.radar_array:
        #     loc = (radar.loc.x, radar.loc.y)
        #     self.radar_loc_plot.addPoints(pos=[loc])

        # self.plot.addItem(self.radar_loc_plot)



        # self.pw_trajectory = self.plot.plot()
        self.sw_trajectory = pg.ScatterPlotItem(
            size=25, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 255, 10))
        self.plot.addItem(self.sw_trajectory)

        # Add radar detection marker plot
        self.det_loc_plot = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 255))


        self.plot.addItem(self.det_loc_plot)

        # Set up plot
        #self.plot.setLimits(yMin=0)
        self.plot.setRange(yRange=[0, self.max_range], xRange=[-self.max_range, self.max_range])
        self.plot.setAspectLocked(True)
            # xMin=-self.max_range, xMax=self.max_range)

        self.plot.setLabel('left', text='Downrange', units='m')
        self.plot.setLabel('bottom', text='Crossrange', units='m')
        self.plot.setTitle('Polar Tracker')

        # Remove extra margins around plot
        self.ci.layout.setContentsMargins(0, 0, 0, 0)

    def update(self):
        '''
        Draw detections on graph.
        '''
        self.det_loc_plot.clear()
        self.sw_trajectory.clear()
        if self.tracker.detections:
            det = self.tracker.detections[0]

            R = det.location.p[0]
            theta = det.location.p[1]

            x = R * np.cos(theta)
            y = R * np.sin(theta)

            if not math.isnan(theta):

                self.det_loc *= 1.0 - self.weight
                self.det_loc += Point(x, y) * self.weight

                print('PolarTrackerWidget(): det_loc', self.det_loc)

                self.trajectory.append(self.det_loc)

            data = np.array([(p[0], p[1]) for p in self.trajectory[-self.trail_length:]])
            self.sw_trajectory.addPoints(pos=data)

            # trajectory_pen = pg.mkPen({'color': "FFFA", 'width': 2})
            # self.pw_trajectory.setData(data, pen=trajectory_pen)

            self.det_loc_plot.addPoints(pos=[self.det_loc])

    def reset(self):
        # self.tracker.reset()
        self.trajectory = []
        self.update()

    # === UTILITY FUNCTIONS ===================================================
    def draw_circle(self, curve, cir, num_pts=100, color="AAFFFF16"):
        '''
        adds a Circle, c, to the plot
        '''
        x_list = []
        y_list = []

        for i in range(num_pts):
            ang = 2 * np.pi * (i / num_pts)
            x = (np.cos(ang) * cir.r) + cir.c.x
            y = (np.sin(ang) * cir.r) + cir.c.y
            x_list.append(x)
            y_list.append(y)

        # append first point to end to 'close' circle
        x_list.append(x_list[0])
        y_list.append(y_list[0])

        x_pts = np.array(x_list)
        y_pts = np.array(y_list)

        curve.setData(x=x_pts, y=y_pts, pen=pg.mkPen(
            {'color': color, 'width': 3}))


    def draw_triangle(self, curve, pts, color="AAFFFF16"):
        """Create triangle object from points."""
        curve.clear()
        for pt in pts:
            curve.append(pt)
        curve.append(pts[0])

    def ppm(self):
        '''
        pixels per meter
        '''
        os = platform.system().lower()
        if os == 'windows':
            pixels = self.frameGeometry().width() - 55.75
            meters = self.plot.vb.viewRange(
            )[0][1] - self.plot.vb.viewRange()[0][0]
        elif os == 'darwin':
            pixels = self.frameGeometry().width() - 55.75
            meters = self.plot.vb.viewRange(
            )[0][1] - self.plot.vb.viewRange()[0][0]
        return pixels / meters
