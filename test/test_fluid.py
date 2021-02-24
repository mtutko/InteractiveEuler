import unittest
import numpy as np
import interactiveeuler.fluid as fl


class ClassesTest(unittest.TestCase):

    def test_Grid(self):
        N = 150
        grid = fl.Grid(N)
        np.testing.assert_allclose(grid.x,
                                   np.linspace(0, 1, N))
        np.testing.assert_allclose(grid.y,
                                   np.linspace(0, 1, N))

    def test_Fluid(self):
        N = 200
        visc = 0.5
        kS = 1.5
        aS = 5.1
        dt = 0.01
        sol = fl.Fluid(N, visc, kS, aS, dt)

        np.testing.assert_almost_equal(sol.visc, visc)
        np.testing.assert_almost_equal(sol.kS, kS)
        np.testing.assert_almost_equal(sol.aS, aS)
        np.testing.assert_almost_equal(sol.dt, dt)


class FluidTest(unittest.TestCase):

    def test_LinInterp_constant(self):
        N = 5
        grid = fl.Grid(N)

        xval = 0.3
        yval = 0.8
        x0 = np.array([xval, yval])

        # test on constant field
        constant = 5.2
        S = constant*np.ones((N, N), dtype=float)
        value1 = fl.LinInterp(S, x0, grid)
        np.testing.assert_almost_equal(value1, constant)

        # test on linear field
        S = grid.X
        value2 = fl.LinInterp(S, x0, grid)
        np.testing.assert_almost_equal(value2, xval)

        # test on bilinear field
        S = grid.X + 2*grid.Y + np.multiply(grid.X, grid.Y)
        value3 = fl.LinInterp(S, x0, grid)
        np.testing.assert_almost_equal(value3, xval + 2*yval + xval*yval)

        # test on quadratic field (interpolation not exact)
        S = np.multiply(grid.X, grid.X)
        value4 = fl.LinInterp(S, x0, grid)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_almost_equal(value4, xval**2)

    def test_TraceParticle(self):
        pass

    def test_Transport(self):
        pass

    def test_addForce(self):
        N = 12
        S0 = 3*np.ones((N, N), dtype=float)
        dt = 0.3
        source = 5.6*np.ones((N, N), dtype=float)

        np.testing.assert_allclose(fl.addForce(S0, source, dt),
                                   S0 + dt*source)


if __name__ == '__main__':
    unittest.main()
