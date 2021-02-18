import unittest
import numpy as np
from interactiveeuler.fluid import (LinInterp,
                                    TraceParticle,
                                    Transport,
                                    addForce)


class FluidTest(unittest.TestCase):

    def test_LinInterp(self):
        pass

    def test_TraceParticle(self):
        pass

    def test_Transport(self):
        pass

    def test_addForce(self):
        N = 12
        S0 = 3*np.ones((N, N), dtype=float)
        dt = 0.3
        source = 5.6*np.ones((N, N), dtype=float)

        np.testing.assert_allclose(addForce(S0, source, dt),
                                   S0 + dt*source)


if __name__ == '__main__':
    unittest.main()
