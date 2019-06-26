"""
Geometry Class.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import numpy as np
from pyratk.datatypes.geometry import Point


class StateVector(object):
    """1-Dimensional state vector object.

    TODO: Allow StateVector to inherit parent object.
    """

    def __init__(self, mat=None, parent=None, axis=None):
        """Initialize vector."""
        # Check for appropriate dimensionality and type
        if type(mat) is not np.ndarray:
            raise TypeError("Argument 'mat' must be of type numpy.ndarray.")
        if mat.shape != (3,):
            raise ValueError("Argument 'mat' must be of shape (3,).")

        self.q = mat
        self.parent = parent
        self.axis = axis

    def __repr__(self):
        """Print StateVector to output stream."""
        return 'pos: {:+6.3}\tvel: {:+6.3}\tacc: {:+6.3}'.format(
            self.p, self.v, self.a)

    def __getitem__(self, val):
        return self.q[val]

    def __setitem__(self, idx, val):
        self.q[idx] = val

    def __len__(self):
        return len(self.q)

    @property
    def p(self):
        """Getter for positional element of state vector."""
        return self.q[0]

    @property
    def v(self):
        """Getter for velocity element of state vector."""
        return self.q[1]

    @property
    def a(self):
        """Getter for acceleration element of state vector."""
        return self.q[2]

    @p.setter
    def p(self, val):
        """Setter for positional element of state vector."""
        self.q[0] = val

    @v.setter
    def v(self, val):
        """Setter for velocity element of state vector."""
        self.q[1] = val

    @a.setter
    def a(self, val):
        """Setter for acceleration for element of state vector."""
        self.q[2] = val


class StateMatrix(object):
    """3-Dimensional state matrix object."""

    coordinate_types = {
        'cartesian': ('x', 'y', 'z'),
        'cylindrical': ('r', 'theta', 'z'),
        'spherical': ('rho', 'theta', 'phi')
    }

    # Valid aliases used for q indexing
    # aliases = {
    #     # q[0] aliases
    #     'x': 'q0',
    #     'r': 'q0',
    #     'rho': 'q0',
    #     # q[1] aliases
    #     'y': 'q1',
    #     'theta': 'q1',
    #     # q[0] aliases
    #     'z': 'q2',
    #     'phi': 'q2'
    # }

    def __init__(self, mat, coordinate_type='cartesian'):
        """Initialize matrix and coordinate type."""
        # Check coordinate type
        if coordinate_type not in self.coordinate_types.keys():
            raise ValueError("Coordinate type must be one of the following:",
                             self.valid_coordinate_types)

        self.coordinate_type = coordinate_type

        # Check for appropriate dimensionality and type
        if type(mat) is not np.ndarray:
            raise TypeError("Argument 'mat' must be of type numpy.ndarray.")

        if mat.shape != (3, 3):
            raise ValueError("Argument 'mat' must be of shape (3, 3).")

        self.q = mat

    # === Alias handling === #
    # def __setattr__(self, name, value):
    #     """Override set attribute method to handle aliases."""
    #     name = self.aliases.get(name, name)
    #     object.__setattr__(self, name, value)
    #
    # def __getattr__(self, name):
    #     """Override set attribute method to handle aliases and error check."""
    #     if name == "aliases":
    #         # recursively calles __getattr__ aliases
    #         raise AttributeError
    #
    #     # Perform coordinate type checking
    #     if self.coordinate_type == 'cartesian':
    #         if name not in self.coordinate_types['cartesian']:
    #             raise AttributeError("Cartesian coordinate has no attribute",
    #                                  name)
    #     elif self.coordinate_type == 'cylindrical':
    #         if name not in self.coordinate_types['cylindrical']:
    #             raise AttributeError("Cylindrical coordinate has no attribute",
    #                                  name)
    #     elif self.coordinate_type == 'spherical':
    #         if name not in self.coordinate_types['spherical']:
    #             raise AttributeError("Spherical coordinate has no attribute",
    #                                  name)
    #
    #     name = self.aliases.get(name, name)
    #     return object.__getattribute__(self, name)

    def __len__(self):
        """Return outer dimension of matrix."""
        return self.q.shape[0]

    def __getitem__(self, idx):
        """Return outer dimension of matrix."""
        return StateVector(self.q[idx])

    def __copy__(self):
        """Return copy of self."""
        return StateMatrix(self.q)

    def __repr__(self):
        """Print StateVector to output stream."""
        if self.coordinate_type == 'cartesian':
            coord_list = self.coordinate_types['cartesian']
        elif self.coordinate_type == 'cylindrical':
            coord_list = self.coordinate_types['cylindrical']
        elif self.coordinate_type == 'spherical':
            coord_list = self.coordinate_types['spherical']

        ret_str = '\n' + ' '*6 + '{:^7}{:^7}{:^7}'.format(*coord_list)

        # loop through all indexes in each state vector
        derivative_list = ('pos', 'vel', 'acc')
        for i in range(3):
            ret_str += '\n{:>6}{:+7.3}{:+7.3}{:+7.3}'.format(
                derivative_list[i], self.x[i], self.y[i], self.z[i]
            )
        return ret_str + '\n'

    def get_state(self, coordinate_type='cartesian', origin=Point()):
        """Return state matrix in specified coordinate system and origin.

        Args:
            coordinate_type (String): Desired coordinate type.
            origin (Point): Desired orign/reference point.

        Raises:
            RuntimeError: If unsupported axis type is provided.

        Returns:
            numpy.ndarray(n,): Numpy array object describing the desired axes,
                in order, of the provided differentiation order (where n is the
                number of axes requested).

        """
        return_list = np.empty(self.q.shape)

        for axis in range(len(return_list)):
            # Add new value to array to be returned.
            axis_value = self.get_axis(axis, coordinate_type, origin)
            return_list[axis] = axis_value

        return np.array(return_list)

    def get_axis(self, axis, coordinate_type='cartesian', origin=Point()):
        """Return desired axis of derivatives wrt origin in coordinate system.

        Args:
            axis (Int): Index of axis to be returned
            coordinate_type (String): Type of coordinate system to return
            origin (Point): 3D Point representing desired origin.

        Returns:
            numpy.ndarray(3,): Requested axis.

        """
        if coordinate_type == 'cartesian':
            return_axis = self.q[axis]

            # Correct for origin offset
            return_axis[0] -= origin.p[axis]

        elif coordinate_type == 'cylindrical':
            raise RuntimeError('Cylindrical is not supported at this time.')

        elif coordinate_type == 'spherical':
            # Relative point (rp), corrected for origin location
            rp = Point(*self.q[:, 0]) - origin

            # Rho axis
            if axis == 0:
                rho = np.empty(self.q.shape[1])
                rho[0] = rp.length
                rho[1] = (rp.x * self.x[1]
                          + rp.y * self.y[1]
                          + rp.z * self.z[1]) / rho[0]
                # TODO: Add acceleration
                rho[2] = 0
                return_axis = rho

            # Phi axis (azimuth)
            if axis == 1:
                phi = np.empty(self.q.shape[1])
                phi[0] = np.arctan2(rp.y, rp.x)
                phi[1] = (rp.x - rp.y) / (rp.x**2 + rp.y**2)
                # TODO: Add acceleration
                phi[2] = 0
                return_axis = phi

            # Theta axis (elevation)
            if axis == 2:
                theta = np.empty(self.q.shape[1])
                theta[0] = np.arccos(rp.z / rp.length)
                theta[1] = (-rp.x**2 + rp.z * rp.x - rp.y**2 + rp.z * rp.y) /\
                    (np.sqrt(1-(rp.z**2 / (rp.length**2))) * rp.length**3)
                # TODO: Add acceleration
                theta[2] = 0
                return_axis = theta
        else:
            raise RuntimeError("'{}' is an unsupported coordinate type."
                               .format(coordinate_type))

        return np.array(return_axis)

    # @property
    # def v(self, coord='cartesian'):
    #     """Return velocity vector."""
    #     return self.q[:, 1]
    #
    # @property
    # def a(self, coord='cartesian'):
    #     """Return acceleration vector."""
    #     return self.q[:, 2]

    # === Coordinates ===
    def get_coordinate_type(self):
        """Return coordinate type.

        Returns:
            String: coordinate type of StateMatrix.

        """
        return self.coordinate_type

    # Cartesian
    @property
    def x(self):
        """First axis of state matrix, q[0]."""
        return StateVector(self.q[0], parent=self)

    @property
    def y(self):
        """Second axis of state matrix, q[1]."""
        return StateVector(self.q[1], parent=self)

    @property
    def z(self):
        """Third axis of state matrix, q[2]."""
        return StateVector(self.q[2], parent=self)

    @x.setter
    def x(self, state_vec):
        """Setter for first axis of state matrix, q[0]."""
        state_type = type(state_vec)
        if state_type is np.ndarray:
            self.q[0] = state_vec
        elif state_type is StateVector:
            self.q[0] = state_vec.q
        else:
            raise TypeError("Argument must be ndarray of shape (3,) or"
                            " StateVector")

    @y.setter
    def y(self, state_vec):
        """Setter for second axis of state matrix, q[1]."""
        state_type = type(state_vec)
        if state_type is np.ndarray:
            self.q[1] = state_vec
        elif state_type is StateVector:
            self.q[1] = state_vec.q
        else:
            raise TypeError("Argument must be ndarray of shape (3,) or"
                            " StateVector")

    @z.setter
    def z(self, state_vec):
        """Setter for third axis of state matrix, q[2]."""
        state_type = type(state_vec)
        if state_type is np.ndarray:
            self.q[2] = state_vec
        elif state_type is StateVector:
            self.q[2] = state_vec.q
        else:
            raise TypeError("Argument must be ndarray of shape (3,) or"
                            " StateVector")
