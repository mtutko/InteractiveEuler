def euler_step(f0, x0, t0, dt):
    """Euler's method step to solve x' = f(x,t) with x(t0) = x0
    and f(x0, t0) = f0.

    USAGE:
        x1 = euler_step(f0, x0, t0, dt)
    """

    x1 = x0 + dt*f0

    return x1





def LinInterp(S, X0):
    """ Linearly interpolates value of field S at point X0
    to scalar value S0. Returns S0. """
    pass


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
    S0 : matrix
        The scalar field, 2d matrix
    source : matrix
        The source field, 2d matrix
    dt : float
        time step

    Returns
    -------
    matrix
        Result of force add calculation
    """

    return S0 + dt*source
