"""
Geometry Class.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import numpy as np
from pyratk.datatypes.geometry import Point


class StateMatrix(object):
    """3-Dimensional state matrix object."""

    coordinate_types = {
        'cartesian': ('x', 'y', 'z'),
        'cylindrical': ('r', 'theta', 'z'),
        'spherical': ('rho', 'theta', 'phi')
    }

    def __init__(self, mat, coordinate_type='cartesian',
                 axis_type='column_major'):
        """Initialize matrix and coordinate type."""
        # TODO: Add row_major axis type
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

    def __len__(self):
        """Return outer dimension of matrix."""
        return self.q.shape[0]

    def __getitem__(self, idx):
        """Return item at index of matrix."""
        return self.q[idx]

    def copy(self):
        """Return copy of self."""
        return StateMatrix(self.q, coordinate_type=self.coordinate_type)

    def __repr__(self):
        """Print StateMatrix to output stream."""
        if self.coordinate_type == 'cartesian':
            coord_list = self.coordinate_types['cartesian']
        elif self.coordinate_type == 'cylindrical':
            coord_list = self.coordinate_types['cylindrical']
        elif self.coordinate_type == 'spherical':
            coord_list = self.coordinate_types['spherical']

        ret_str = '\n' + ' '*6 + ' {:^9} {:^9} {:^9}'.format(*coord_list)

        # loop through all indexes in each state vector
        derivative_list = ('pos', 'vel', 'acc')
        for i in range(3):
            ret_str += '\n{:>6} {:>+9.3} {:>+9.3} {:>+9.3}'.format(
                derivative_list[i], self.x[i], self.y[i], self.z[i]
            )
        return ret_str + '\n'

    def get_state(self, coordinate_type='cartesian', origin=Point()):
        """Return StateMatrix wrt origin in desired coordinate system.

        Args:
            coordinate_type (String): Type of coordinate system to return
            origin (Point): 3D Point representing desired origin.

        Returns:
            StateMatrix: New StateMatrix with desired origin and coordinte type.

        """
        if coordinate_type == self.coordinate_type and origin == Point():
            # If no change is needed, return self state matrix
            return_matrix = self.q.copy()

        elif coordinate_type == 'cartesian':
            # First convert to cartesian
            if self.coordinate_type == 'cartesian':
                return_matrix = self.q.copy()
            elif self.coordinate_type == 'cylindrical':
                return_matrix = np.empty(self.q.shape)

                # State vectors
                rho = self.q[0]
                phi = self.q[1]
                z = self.q[2]

                # Zeroth derivative
                return_matrix[:, 0] = np.array([
                    rho[0] * np.cos(phi[0]),
                    rho[0] * np.sin(phi[0]),
                    z[0]
                ])

                # First derivative
                return_matrix[:, 1] = np.array([
                    rho[1] * np.cos(phi[0]) - rho[0] * phi[1] * np.sin(phi[0]),
                    rho[1] * np.sin(phi[0]) + rho[0] * phi[1] * np.cos(phi[0]),
                    z[1]
                ])

                # Second derivative
                return_matrix[:, 2] = np.array([
                    - phi[1]**2 * rho[0] * np.cos(phi[0])
                    - 2 * phi[1] * rho[1] * np.sin(phi[0])
                    - phi[2] * rho[0] * np.sin(phi[0])
                    + rho[2] * np.cos(phi[0]),
                    - phi[1]**2 * rho[0] * np.sin(phi[0])
                    + 2 * phi[1] * rho[1] * np.cos(phi[0])
                    + phi[2] * rho[0] * np.cos(phi[0])
                    + rho[2] * np.sin(phi[0]),
                    z[2]
                ])

            else:
                raise RuntimeError("'{}' is an unsupported coordinate type to "
                                   "convert to cartesian."
                                   .format(self.coordinate_type))

            # Correct for origin offset
            return_matrix[:, 0] -= origin.p

        elif coordinate_type == 'cylindrical':
            # Relative point (rp), corrected for origin location
            rp = self.get_state('cartesian', origin)

            # Radius axis (Rho)
            rho = np.empty(rp.shape[1])
            rho[0] = np.sqrt(rp[0][0]**2 + rp[1][0]**2)
            rho[1] = (rp[0][0] * rp[0][1] + rp[1][0] * rp[1][1]) / rho[0]
            # TODO: Add acceleration
            rho[2] = 0

            # Azimuth axis (Phi)
            phi = np.empty(rp.shape[1])
            phi[0] = np.arctan2(rp[1][0], rp[0][0])
            phi[1] = (rp[0][0] - rp[1][0]) / (rp[0][0]**2 + rp[1][0]**2)
            # TODO: Add acceleration
            phi[2] = 0

            return_matrix = np.array((rho, phi, rp[2][0]))

        elif coordinate_type == 'spherical':
            # Relative point (rp), corrected for origin location
            rp = self.get_state('cartesian', origin)

            # Radius axis (Rho)
            rho = np.empty(rp.shape[1])
            rho[0] = np.sqrt(np.sum(rp[:, 0]**2))
            rho[1] = (rp[0][0] * rp[0][1]
                      + rp[1][0] * rp[1][1]
                      + rp[2][0] * rp[2][1]) / rho[0]
            # TODO: Add acceleration
            rho[2] = 0

            # Azimuth axis (Phi)
            phi = np.empty(rp.shape[1])
            phi[0] = np.arctan2(rp[1][0], rp[0][0])
            phi[1] = (rp[0][0] * rp[1][1] - rp[0][1] * rp[1][0]) \
                / (rp[0][0]**2 + rp[1][0]**2)
            # TODO: Add acceleration
            phi[2] = 0

            # Elevation axis (Theta)
            theta = np.empty(rp.shape[1])
            theta[0] = np.arccos(rp[2][0] / rho[0])
            theta[1] = (-rp[0][0]**2 + rp[2][0] * rp[0][0]
                        - rp[1][0]**2 + rp[2][0] * rp[1][0])\
                / (np.sqrt(1-(rp[2][0]**2 / (rho[0]**2))) * rho[0]**3)
            # TODO: Add acceleration
            theta[2] = 0

            return_matrix = np.array((rho, phi, theta))
        else:
            raise RuntimeError("'{}' is an unsupported coordinate type to "
                               "convert to."
                               .format(coordinate_type))

        return StateMatrix(return_matrix, coordinate_type=coordinate_type)

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

    @property
    def shape(self):
        return self.q.shape

    # Cartesian
    @property
    def x(self):
        """First axis of state matrix, q[0]."""
        return self.q[0]

    @property
    def y(self):
        """Second axis of state matrix, q[1]."""
        return self.q[1]

    @property
    def z(self):
        """Third axis of state matrix, q[2]."""
        return self.q[2]

    @x.setter
    def x(self, state_vec):
        """Setter for first axis of state matrix, q[0]."""
        state_type = type(state_vec)
        if state_type is np.ndarray:
            self.q[0] = state_vec
        else:
            raise TypeError("Argument must be ndarray of shape (3,)")

    @y.setter
    def y(self, state_vec):
        """Setter for second axis of state matrix, q[1]."""
        state_type = type(state_vec)
        if state_type is np.ndarray:
            self.q[1] = state_vec
        else:
            raise TypeError("Argument must be ndarray of shape (3,)")

    @z.setter
    def z(self, state_vec):
        """Setter for third axis of state matrix, q[2]."""
        state_type = type(state_vec)
        if state_type is np.ndarray:
            self.q[2] = state_vec
        else:
            raise TypeError("Argument must be ndarray of shape (3,)")
