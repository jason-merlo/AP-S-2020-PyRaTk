# -*- coding: utf-8 -*-
"""
Tracker2D Widget Class.

Contains parametric graph capable of plotting a tracked object's path.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import pyqtgraph as pg          # Used for RadarWidget superclass
import numpy as np              # Used for numerical operations
import platform                 # Get OS for DPI scaling
from pyratk.datatypes.geometry import Point, Circle


class Tracker2dWidget(pg.GraphicsLayoutWidget):
    def __init__(self, tracker, xRange=[-0.10, 0.10], yRange=[-0.10, 0.10],
                 trail=1):
        super(Tracker2dWidget, self).__init__()

        # Copy arguments to member variables
        self.tracker = tracker
        self.xRange = xRange
        self.yRange = yRange
        self.trail = trail

        # Add plots to layout
        self.plot = self.addPlot()

        # Add radar location markers
        self.radar_loc_plot = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 255))

        for radar in self.tracker.radar_array:
            loc = (radar.loc.x, radar.loc.y)
            self.radar_loc_plot.addPoints(pos=[loc])

        self.plot.addItem(self.radar_loc_plot)

        # Add radar detection radius circles
        self.radar_rad_plots = []
        for i in range(len(self.tracker.radar_array)):
            circle = pg.PlotCurveItem()

            # Add initial radii
            radar = self.tracker.radar_array[i]
            self.draw_circle(circle, Circle(radar.loc, 0))

            # Add to list to keep track
            self.radar_rad_plots.append(circle)

            # Add to plot
            self.plot.addItem(circle)

        # Add triangle for tracker
        self.tracker_triangle = pg.PlotCurveItem()
        self.plot.addItem(self.tracker_triangle)

        # Set up plot
        self.plot.setRange(xRange=self.xRange, yRange=self.yRange)
        self.pw = self.plot.plot()
        self.plot.setLabel('left', text='Distance-Y', units='m')
        self.plot.setLabel('bottom', text='Distance-X', units='m')
        self.plot.setTitle('Tracker2D')

        self.plot.setAspectLocked(True)

        # Remove extra margins around plot
        self.ci.layout.setContentsMargins(0, 0, 0, 0)

    def update(self):
        '''
        Draws track trail and radar circles on tracker2d graph
        '''
        # === Draw trace line =========
        data = self.tracker.ts_location.data[-self.trail:, :, 0]
        data = np.array([(p[0], p[1]) for p in data])
        self.pw.setData(data, pen=pg.mkPen({'color': "FFF", 'width': 2}))

        # === Draw tracker triangle ===
        # triangle_x = [p.x for p in self.tracker.best_triangle.points]
        # triangle_y = [p.y for p in self.tracker.best_triangle.points]
        # triangle_x.append(triangle_x[0])
        # triangle_y.append(triangle_y[0])
        # print(triangle)
        # try:
        #     self.tracker_triangle.setData(
        #         triangle_x, triangle_y, pen=pg.mkPen({'color': "F00", 'width': 4}))
        # except Exception as e:
        #     print('Tracker triangle failure:', e)

        # === Draw circles ============
        color_pos = (160, 255, 255)
        color_neg = (255, 160, 160)

        for i, pt in enumerate(self.radar_rad_plots):
            radar = self.tracker.radar_array[i]
            vel = radar.drho / 10   # scaled to fit on screen;
            #                       # Velocity of highest intensity return
            try:
                rad_vec = Point(*self.tracker.ts_location[-1, :, 0]) - radar.loc
                rad_vec.z = 0.0
                rad = rad_vec.length
            except Exception as e:
                # print('Tracker widget exception:', e)
                rad = 0
            # visualizer = 'vel_rad'
            visualizer = 'vel_rad'

            vel = -vel * 2
            if vel < 0:
                vel = -vel
                if visualizer == 'abs_rad':
                    color = (*color_neg, 128)
                    self.draw_circle(pt, Circle(radar.loc, rad), color=color)
                elif visualizer == 'vel_rad':
                    color = (*color_neg, 128)
                    self.draw_circle(pt, Circle(radar.loc, vel), color=color)
            else:
                if visualizer == 'abs_rad':
                    color = (*color_pos, 128)
                    self.draw_circle(pt, Circle(radar.loc, rad), color=color)
                elif visualizer == 'vel_rad':
                    color = (*color_pos, 128)
                    self.draw_circle(pt, Circle(radar.loc, vel), color=color)

    def reset(self):
        # self.tracker.reset()
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


# class CircleItem(pg.GraphicsObject):
#     def __init__(self, circle, color="AAFFFFFF"):
#         pg.GraphicsObject.__init__(self)
#         self.circle = circle
#         self.generatePicture(color)
#
#     def generatePicture(self, color):
#         """Pre-computing QPicture object allows paint() to run faster."""
#         self.picture = pg.QtGui.QPicture()
#         p = pg.QtGui.QPainter(self.picture)
#         p.setPen(pg.mkPen({'color': color, 'width': 10}))
#         p.setBrush(pg.mkBrush('g'))
#         p.drawEllipse(self.circle.c.x, self.circle.c.y,
#                       self.circle.r * 2, self.circle.r * 2)
#         print(self.circle)
#         p.end()
#         self.paint(p)
#
#     def setData(self, circle, color="AAFFFFFF"):
#         self.flagHasData = True
#         self.circle = circle
#         self.generatePicture(color)
#
#     def paint(self, p, *args):
#         """PyQtGraph will call this."""
#         p.drawPicture(0, 0, self.picture)
#
#     def boundingRect(self):
#         """Indicate the entire area that will be drawn on."""
#         return pg.QtCore.QRectF(self.picture.boundingRect())
