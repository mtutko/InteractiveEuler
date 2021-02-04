

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
    """ Adds force (via "source") to field S0 and multiplies
    by dt. Returns updated field S1. """
    pass
