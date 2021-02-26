from scipy.interpolate import RegularGridInterpolator
import numpy as np

# comment out for production code
import matplotlib.pyplot as plt


class Grid():
    """2D Grid for computational domain.

    This grid is a square [0,1]^2 domain that includes the endpoints.

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
    """Fluid and Scalar solutions and parameters.

    The fluid is assumed to only need 2 update functions and also is assumed
    to contain one scalar compound. Additionally, due to the use of the Stable
    Fluids algorithm (see Stam paper) we do not store the pressure solution,
    but rather provide a method to compute it.

    Note
    ----
    We need to add a method to compute the pressure.

    Parameters
    ----------
    N : int, optional
        Number of nodes along each axis (the default value is 100).
    visc : float, optional
        Viscosity of the fluid (the default value is 1).
    kS : float, optional
        Diffusion rate of scalar (the default value is 1).
    aS : float, optional
        Dissipation rate of scalar (the default value is 1).
    dt : float, optional
        Time step for numerical solution (the default value is 0.01).

    Attributes
    ----------
    U0 : Velocity
        Velocity solution at previous step.
    U1 : Velocity
        Velocity solution at current step.
    S0 : ndarray(dtype=float, ndim=2)
        Scalar solution at previous step.
    S1 : ndarray(dtype=float, ndim=2)
        Scalar solution at current step.
    visc : float
        Viscosity of the fluid.
    kS : float
        Diffusion rate of scalar.
    aS : float
        Dissipation rate of scalar.
    dt : float
        Time step for numerical solution.

    """
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
        self.U = np.zeros((N, N), dtype=float)
        self.V = np.zeros((N, N), dtype=float)


def euler_step(f0, x0, dt):
    """One step of Euler's method for solving ODEs.

    Solves x' = f(x,t) with x(0) = x0 and f(x0, 0) = f0.

    Example:
        x1 = euler_step(f0, x0, dt)
    """

    x1 = x0 + dt*f0

    return x1


def LinInterp(S, x0, grid):
    """ Linearly interpolates value of field S at point x0
    to scalar value S0. Returns S0. """

    interp_func = RegularGridInterpolator((grid.x, grid.y), S,
                                          bounds_error=False,
                                          fill_value=0)
    return interp_func(x0)


def TraceParticle(x, y, U, V, dt):
    """ Uses RK2 to find point (x0, y0) going backwards (-dt)
    along velocity field (U,V) from the point (x,y).

    Note
    ----
    Right now it uses euler's method, not RK2, and it also only takes in floats
    for U and V instead of fields. """

    x0 = euler_step(U, x, -dt)
    y0 = euler_step(V, y, -dt)

    return np.array([x0, y0])


def Transport(grid, S0, U, dt):
    """ Advects scalar field S0 for dt time along velocity
    field U. Returns new scalar field S1. """

    S1 = np.ones_like(S0)       # this function can probably be improved
    for index, _ in np.ndenumerate(S0):
        xval = grid.x[index[0]]
        yval = grid.y[index[1]]
        x0 = TraceParticle(xval, yval, U.U[index], U.V[index], dt)
        S1[index] = LinInterp(S0, x0, grid)

    return S1


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
