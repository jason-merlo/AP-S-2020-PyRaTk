"""
Geometry Class.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import numpy as np


class Point(object):
    def __init__(self, *args):
        self.p = args

    @property
    def p(self):
        return (self.x, self.y, self.z)

    @p.setter
    def p(self, *args):
        # Defult to origin if zero arguments provided
        self.x = 0
        self.y = 0
        self.z = 0
        alen = len(args)
        # if argument is tuple
        if alen == 1:
            if len(args[0]) != 0:
                try:
                    self.x = args[0][0]
                    self.y = args[0][1]
                    # if tuple is 3D
                    if len(args[0]) == 3:
                        self.z = args[0][2]
                except TypeError:
                    raise TypeError('Point expects tuple, or two numbers, '
                                    'or three numbers; got', type(args[0]))
        # if argument is 2D
        elif alen == 2:
            self.x = args[0]
            self.y = args[1]
        # if argument is 3D
        elif alen == 3:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]
        elif alen != 0:
            raise TypeError('Point expects at most three arguments, got', alen)

    @property
    def r(self, a):
        """Return `R` in spherical coordinates relative to `a`.

        Args:
            a (Point): Point to compute radius relative to.

        Returns:
            float: Distance from Point `a`.

        """
        return self.distance(a)

    @property
    def phi(self):
        """
        Return `phi` in spherical coordinates in radians.

        `phi` is defined as the angle between the x-axis and the line which
        extends through the point projected onto the xy-plane.
        """
        return np.arctan2(self.y, self.x)

    @property
    def theta(self):
        """
        Return `theta` in spherical coordinates in radians.

        `theta` is defined as the angle between the z-axis and the line which
        extends through the point.
        """
        xy_dist = np.sqrt(self.x**2, self.y**2)
        return np.arctan2(xy_dist, self.z)

    @property
    def sph(self):
        """Return a Point representing spherical coordinates."""
        return Point(self.r, self.phi, self.theta)

    def distance(self, a=None):
        """Find distance between self and another Point."""
        # If no arguments provided, compare to origin
        if a is None:
            a = Point(0, 0)
        dx = self.x - a.x
        dy = self.y - a.y
        dz = self.z - a.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    # def angle(self, a=None):
    #     """Find angle between self and another point in radians.
    #
    #     The angle computed is relative to the point provided (where the point
    #     provided is 0 radians).  If not point is provided, the origin is used
    #     as a reference.
    #
    #     Args:
    #         a (Point): Reference Point used to compute angle to.
    #
    #     Returns:
    #         numpy.ndarray(3,): Angle between two points (r, phi, theta) in
    #             radians.
    #
    #     """
    #     if a is None:
    #         a = Point(0, 0)
    #     r =
    #     return np.arctan2()

    def normalize(self):
        '''
        Convert Point into a vector of length 1
        '''
        len = self.length
        if len != 0:
            self.x /= len
            self.y /= len
            self.z /= len

    @property
    def length(self):
        '''
        Returns the distance of the point from the origin
        '''
        a = self.x**2 + self.y**2 + self.z**2
        return np.sqrt(a)

    def copy(self):
        return Point(*self.p)

    def __getitem__(self, i):
        item = None
        if i == 0:
            item = self.x
        elif i == 1:
            item = self.y
        elif i == 2:
            item = self.z
        else:
            raise IndexError("Index cannot exceed 2 for Point(x,y,z)")
        return item

    def __len__(self):
        return 3

    def __add__(self, a):
        ret = Point()
        if isinstance(a, Point):
            ret.x = self.x + a.x
            ret.y = self.y + a.y
            ret.z = self.z + a.z
        else:
            ret.x = self.x + a
            ret.y = self.y + a
            ret.z = self.z + a

        return ret

    def __sub__(self, a):
        ret = Point()
        if isinstance(a, Point):
            ret.x = self.x - a.x
            ret.y = self.y - a.y
            ret.z = self.z - a.z
        else:
            ret.x = self.x - a
            ret.y = self.y - a
            ret.z = self.z - a

        return ret

    def __mul__(self, a):
        ret = Point()
        if isinstance(a, Point):
            ret.x = self.x * a.x
            ret.y = self.y * a.y
            ret.z = self.z * a.z
        else:
            ret.x = self.x * a
            ret.y = self.y * a
            ret.z = self.z * a

        return ret

    def __truediv__(self, a):
        ret = Point()
        if isinstance(a, Point):
            ret.x = self.x / a.x
            ret.y = self.y / a.y
            ret.z = self.z / a.z
        else:
            ret.x = self.x / a
            ret.y = self.y / a
            ret.z = self.z / a

        return ret

    def __repr__(self):
        return '({:+11.7f}, {:+11.7f}, {:+11.7f})'.format(*self.p)


class Circle(object):
    def __init__(self, *args):
        '''
        Circle class, holds Point and radius, also finds intersections

        Args:
            tuple
                (Point(), r)
            (or)
            Point()
                Point object containing center of circle
            r
                Number indicating radius of circle
        '''
        self.circle = args

    @property
    def circle(self):
        return (self.c, self.r)

    @circle.setter
    def circle(self, *args):
        # Default to unit circle if zero arguments provided
        self.c = Point(0, 0)
        self.r = 1

        alen = len(args)

        if alen == 1:
            if len(args[0]) != 0:
                try:
                    self.c = args[0][0]
                    self.r = args[0][1]
                except TypeError:
                    raise TypeError(
                        'Circle expects tuple, a Point and number, '
                        'or three numbers; got', type(args[0]))
        elif alen == 2:
            self.c = args[0]
            self.r = args[1]
        elif alen == 3:
            self.c = Point(args[0], args[1])
            self.r = args[2]
        elif alen != 0:
            raise TypeError(
                'Circle expects at most three arguments, got', alen)

    def intersections(self, c):
        '''
        Finds the intersections of two circles, or the mid-point between
        their circumferences

        Based on the algorithm found at:
        http://paulbourke.net/geometry/circlesphere/

        Args:
            c
                the circle to calculate intersections with

        Returns:
            list of one or two Points
        '''
        # distance between centers
        dist = self.c.distance(c.c)

        # One circle is contained within the other
        if dist < np.absolute(self.r - c.r):
            # print("WARNING: Circles are contained within each other")
            if self.r < c.r:
                c1 = self
                c2 = c
            else:
                c1 = c
                c2 = self
            # get unit vector towards smaller circle within larger circle
            dir = c2.c - c1.c
            dir.normalize()
            # ditance between circumferences
            dc = c1.r - c2.r - dist
            mp_dist = dist + c2.r + dc

            result = [dir * mp_dist]
        # Circles intersect
        elif dist < self.r + c.r and self.r != 0 and c.r != 0:
            # distance to midpoint within both circumferences
            mp_dist = (self.r**2 - c.r**2 + dist**2) / (2 * dist)
            # distance to intersection from midpoint
            d2 = self.r**2 - mp_dist**2
            if (d2 < 0):
                print("WARNING: d2 is negative: ", d2, "*" * 50)
            height = np.sqrt(abs(d2))
            # Point object between both circumferences
            d_vec = (c.c - self.c)
            midpoint = Point()
            midpoint.x = self.c.x + (mp_dist * d_vec.x) / dist
            midpoint.y = self.c.y + (mp_dist * d_vec.y) / dist
            # midpoint = self.c + (mp_dist * d_vec) / dist
            # intersections
            P1 = Point()
            P2 = Point()
            P1.x = midpoint.x + (height * d_vec.y) / dist
            P1.y = midpoint.y - (height * d_vec.x) / dist
            P2.x = midpoint.x - (height * d_vec.y) / dist
            P2.y = midpoint.y + (height * d_vec.x) / dist
            result = [P1, P2]
        # circles do not intersect, and are not inside one another
        else:
            # print("WARNING: No intersection found, using midpoint")
            # Calculate unit vector in direction of second circle
            dir = c.c - self.c
            dir.normalize()
            dc = (dist - self.r - c.r) / 2
            mp_dist = self.r + dc
            result = [self.c + (dir * mp_dist)]
            # print('c1', self, '\nc2', c, '\nmp', result)

        return result

    def __repr__(self):
        return 'Circle:\n\tC: {:}\tR: {:+6.3}'.format(self.c, self.r)


class Triangle(object):
    def __init__(self, *args):
        self.points = args

    @property
    def points(self):
        return (self.p)

    @points.setter
    def points(self, *args):
        self.p = (Point(), Point(), Point())

        alen = len(args)
        if alen == 1:
            if len(args[0]) != 0:
                try:
                    self.p = (args[0][0], args[0][1], args[0][2])
                except TypeError:
                    raise TypeError('Triangle expects tuple, or three numbers;'
                                    ' got', type(args[0]))
        elif alen == 3:
            self.p = (args[0], args[1], args[2])
        else:
            raise TypeError(
                'Triangle expects at most three arguments, got', alen)

    @property
    def centroid(self):
        centroid = Point()
        for point in self.points:
            centroid += point
        return centroid / 3

    @property
    def area(self):
        # Determinant method
        p = self.points
        a = np.array([[p[0].x, p[1].x, p[2].x],
                      [p[0].y, p[1].y, p[2].y],
                      [1, 1, 1]])
        area = 0.5 * abs(np.linalg.det(a))
        return area

    @property
    def circumference(self):
        p = self.points
        c = p[0].distance(p[1]) + p[1].distance(p[2]) + p[2].distance(p[0])
        return c

    def __repr__(self):
        return 'Triangle:\n\t{:}\n\t{:}\n\t{:}'.format(*self.points)
