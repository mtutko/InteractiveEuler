from scipy.interpolate import RegularGridInterpolator
import numpy as np

# comment out for production code
import matplotlib.pyplot as plt


class Grid():
    """2D Grid for computational domain.

    This grid is a square [0,1]^2 domain that includes the endpoints.

    Note
    ----
    Do not include the `self` parameter in the ``Parameters`` section.

    Parameters
    ----------
    N : int
        Number of nodes along each axis.

    Attributes
    ----------
    x : ndarray(dtype=float, ndim=1)
        1D array of points in x-coordinate direction.
    y : ndarray(dtype=float, ndim=1)
        1D array of points in y-coordinate direction.
    X : ndarray(dtype=float, ndim=2)
        2D array of points representing x-coordinate; generated from meshgrid.
    Y : ndarray(dtype=float, ndim=2)
        2D array of points representing y-coordinate; generated from meshgrid.

    """
    def __init__(self, N):
        self.x = np.linspace(0, 1, num=N)
        self.y = np.linspace(0, 1, num=N)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')


class Fluid():
    def __init__(self, N=100, visc=1.0, kS=1.0, aS=1.0, dt=0.01):
        # fields
        self.U0 = Velocity(N)
        self.U1 = Velocity(N)
        self.S0 = np.zeros((N, N), dtype=float)
        self.S1 = np.zeros((N, N), dtype=float)

        # physical parameters
        self.visc = visc
        self.kS = kS
        self.aS = aS

        # numerical parameters
        self.dt = dt


class Velocity():
    def __init__(self, N):
        self.U = np.ones((N, N), dtype=float)
        self.V = np.ones((N, N), dtype=float)


def euler_step(f0, x0, t0, dt):
    """One step of Euler's method for solving ODEs.

    Solves x' = f(x,t) with x(t0) = x0 and f(x0, t0) = f0.

    Example:
        x1 = euler_step(f0, x0, t0, dt)
    """

    x1 = x0 + dt*f0

    return x1


def LinInterp(S, X0, grid):
    """ Linearly interpolates value of field S at point X0
    to scalar value S0. Returns S0. """

    interp_func = RegularGridInterpolator((grid.x, grid.y), S)
    return interp_func(X0)


def TraceParticle(X, U, dt):
    """ Uses RK2 to find point X0 going backwards (-dt)
    along velocity field U from the point X. Returns X0 """
    pass


def Transport(S0, U, dt):
    """ Advects scalar field S0 for dt time along velocity
    field U. Returns new scalar field S1. """
    pass


def addForce(S0, source, dt):
    """Adds force (w1) during Stable Fluid algorithm.

    Adds force (via "source") to field S0 and multiplies by dt. Returns
    updated field S1.

    Parameters
    ----------
    S0 : ndarray(dtype=float, ndim=2)
        The scalar field, 2d matrix
    source : ndarray(dtype=float, ndim=2)
        Source field, 2d matrix
    dt : float
        Time step

    Returns
    -------
    matrix
        Result of force add calculation
    """

    return S0 + dt*source


# These functions are for development only. They should be commented out
# when code goes to production
def plot_fluid(grid, solution):
    X, Y = np.meshgrid(grid.x, grid.y)

    plt.contourf(X, Y, solution.S1, cmap='RdGy')
    plt.colorbar()
    plt.quiver(X, Y, solution.U1.U, solution.U1.V)
    plt.title("Scalar")
    plt.show()
