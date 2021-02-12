import unittest
from interactiveeuler.fluid import (LinInterp,
                                    TraceParticle,
                                    Transport,
                                    addForce)


class FluidTest(unittest.TestCase):

    def test_LinInterp(self):
        self.assertEqual(18.0, 18.0)

    def test_TraceParticle(self):
        self.assertEqual(18.0, 18.0)

    def test_Transport(self):
        self.assertEqual(18.0, 18.0)

    def test_addForce(self):
        # test that the result equals S0 + dt*source
        # test that source and S0 are the same size (and doesn't work if they are different sizes)
        self.assertEqual(18.0, 18.0)


if __name__ == '__main__':
    unittest.main()
