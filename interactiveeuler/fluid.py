from scipy.interpolate import RegularGridInterpolator
import numpy as np

# comment out for production code
import matplotlib.pyplot as plt


class StabilityError(Exception):
    pass


class Grid():
    """2D Grid for computational domain.

    This grid is a square [0,1]^2 domain that includes the endpoints.

    Parameters
    ----------
    N : int, optional
        Number of nodes along each axis (the default value is 100).

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
    def __init__(self, N=100):
        self.x = np.linspace(0, 1, num=N)
        self.y = np.linspace(0, 1, num=N)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.dx = 1 / (N-1)


class Fluid():
    """Class containing Fluid solutions (Velocity and Pressure) and
    parameters (viscosity).

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

    Attributes
    ----------
    U0 : Velocity
        Velocity solution at previous step.
    U1 : Velocity
        Velocity solution at current step.
    P0 : ndarray(dtype=float, ndim=2)
        Pressure solution at previous step.
    P1 : ndarray(dtype=float, ndim=2)
        Pressure solution at current step.
    visc : float
        Viscosity of the fluid.

    """
    def __init__(self, N=100, visc=1.0):
        # fields
        self.U0 = Velocity(N)
        self.U1 = Velocity(N)
        self.P0 = np.zeros((N, N), dtype=float)
        self.P1 = np.zeros((N, N), dtype=float)

        # physical parameters
        self.visc = visc


class Simulation:
    """Class containing parameters relevant to the simulation.

    Parameters
    ----------
    dt : float, optional
        Time step for numerical solution (the default value is 0.01).
    Tfinal : dt, optional
        Final simulation time (the default value is 1.0)
    Attributes
    ----------
    dt : float
        Time step for numerical solution.
    Tfinal : float
        Final simulation time.

    """
    def __init__(self, dt=0.01, Tfinal=1.0):
        self.dt = dt
        self.Tfinal = Tfinal


class Velocity():
    """Class for Velocity solution.

    Parameters
    ----------
    N : int
        Number of nodes along each axis.

    Attributes
    ----------
    U : ndarray(dtype=float, ndim=2)
        Horizontal velocity field.
    V : ndarray(dtype=float, ndim=2)
        Vertical velocity field.

    """
    def __init__(self, N):
        self.U = np.zeros((N, N), dtype=float)
        self.V = np.zeros((N, N), dtype=float)


class Scalar():
    """Class for Scalar solution.

    Parameters
    ----------
    N : int, optional
        Number of nodes along each axis (default value is 100).
    k : float, optional
        Diffusion coefficient for this scalar (default value is 1).
    name: string, optional
        Name of scalar variable (default value is "scalar").

    Attributes
    ----------
    U : ndarray(dtype=float, ndim=2)
        Scalar solution defined on grid.
    k : float
        Diffusion coefficient for this field.
    name: string
        Name of scalar variable.

    """
    def __init__(self, N=100, k=1, name="scalar"):
        self.U = np.zeros((N, N), dtype=float)
        self.k = k
        self.name = name


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


def Diffuse(U0, diff_coef, dt, dx,
            utop=0, ubottom=0, uleft=0, uright=0):
    """Updates the field A0 subject to the PDE

    A_t = diff_coef nabla^2 A

    Uses a straightforward finite difference approximation,
    where the second derivative is approximated by a center-
    difference scheme. This scheme is explicit (nice) but
    also only numerically stable when dt < dx^2/(4*diff_coef)
    (not nice). This scheme should be replaced with Crank-
    Nicholson. If stability criterion is violated, a StabilityError
    exception is raised.

    Note
    ----
    1. Right now this is done element-wise, which is very slow
    in python. Better to vectorize.

    2. This code expects constant dirichlet boundary conditions.

    Parameters
    ----------
    A0 : ndarray(dtype=float, ndim=2)
        The scalar field, 2d matrix
    diff_coef : float
        Diffusion coefficient
    dt : float
        Discretized time step
    dx : float
        Discretized space step
    utop : float, optional
        Top boundary condition (the default value is zero).
    ubottom : float, optional
        Bottom boundary condition (the default value is zero).
    uleft : float, optional
        Left boundary condition (the default value is zero).
    uright : float, optional
        Right boundary condition (the default value is zero).

    Returns
    -------
    U1 : ndarray(dtype=float, ndim=2)
        Updated scalar solution.
    """
    U1 = np.zeros_like(U0)
    N, M = U0.shape

    # check for stability
    if (dt > dx**2/(4*diff_coef)):
        raise StabilityError("dt > dx^2/(4*diff_coef! Simulation is unstable."
                             "Select smaller time step.")

    # diffuse interior
    alpha = diff_coef*dt/dx**2
    for ti in range(1, N-1):
        for tj in range(1, M-1):
            U1[ti, tj] = U0[ti, tj] + alpha*(U0[ti + 1, tj] +
                                             U0[ti - 1, tj] +
                                             U0[ti, tj + 1] +
                                             U0[ti, tj - 1] -
                                             4*U0[ti, tj]
                                             )

    # apply boundary conditions
    U1[N-1, :] = ubottom
    U1[0, :] = utop
    U1[:, 0] = uleft
    U1[:, M-1] = uright

    U1[N-1, M-1] = 0.5*(ubottom + uright)  # bottomright
    U1[0, M-1] = 0.5*(utop + uright)  # topright
    U1[N-1, 0] = 0.5*(ubottom + uleft)  # bottomleft
    U1[0, 0] = 0.5*(utop + uleft)  # topleft

    return U1


def VStep(grid, fluid, F, dt):
    # add force to all values of velocity
    fluid.U1 = addForce(fluid.U0, F, dt)

    # transport all values of velocity
    fluid.U1 = Transport(grid, fluid.U1, fluid.U0, dt)

    # diffuse all values of velocity

    # project all values of velocity

    pass


def Sstep(grid, sim, scalar, fluid, source, dt):
    # add force to all values of scalar
    scalar.S1 = addForce(scalar.S0, source, dt)

    # transport all values of scalar
    scalar.U1 = Transport(grid, scalar.S1, fluid.U0, dt)

    # diffuse all values of scalar
    scalar.U = Diffuse(scalar.U, scalar.k, sim.dt, grid.dx)

    # dissipate all values of scalar

    pass


# ----------------------------------------------------------------------
# These functions are for development only. They should be commented out
# when code goes to production
# ----------------------------------------------------------------------
def plot_fluid(grid, solution):
    plt.contourf(grid.X, grid.Y, solution.P1, cmap='RdGy')
    plt.colorbar()
    plt.quiver(grid.X, grid.Y, solution.U1.U, solution.U1.V)
    plt.title("Pressure")
    plt.show()


def plot_scalar(grid, solution, time):
    # plt.contourf(grid.X, grid.Y, solution.U)
    plt.imshow(solution.U)
    plt.colorbar()
    plt.title(f"{solution.name}, t={time}")
    plt.show()


import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


def animate_it():
    N = 50
    dt = 0.0001
    diff_coef = 1.0
    Tfinal = 1.0
    scalar_name = "Temperature"

    grid = Grid(N)
    sim = Simulation(dt, Tfinal)

    # boundary conditions
    utop = 0
    ubottom = 0
    uleft = 0
    uright = 1

    temp = Scalar(N, diff_coef, scalar_name)

    t = 0
    while t < sim.Tfinal:
        temp.U = Diffuse(temp.U, temp.k, sim.dt, grid.dx,
                         utop=utop, ubottom=ubottom,
                         uleft=uleft, uright=uright)
        t += sim.dt
        print(f"Simulation time t={t} / {sim.Tfinal}", end="\r")

    plot_scalar(grid, temp, t)

    print("Simulation done")
