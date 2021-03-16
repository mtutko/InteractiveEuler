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
        sol = fl.Fluid(N, visc)

        np.testing.assert_almost_equal(sol.visc, visc)


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
        x0 = 1.3
        y0 = 4.1
        dt = 0.3

        # zero velocity test
        u1 = 0.0
        v1 = 0.0
        x1, y1 = fl.TraceParticle(x0, y0, u1, v1, dt)
        np.testing.assert_almost_equal(x1, x0)
        np.testing.assert_almost_equal(y1, y0)

        # constant field test
        u2 = 1.2
        v2 = -4.1
        x2, y2 = fl.TraceParticle(x0, y0, u2, v2, dt)
        np.testing.assert_almost_equal(x2, x0 - dt*u2)
        np.testing.assert_almost_equal(y2, y0 - dt*v2)

    def test_Transport(self):
        N = 5
        grid = fl.Grid(N)
        dt = 0.3
        U = fl.Velocity(N)

        # transport constant field with zero velocity
        const = 3.1
        S0 = const*np.ones((N, N), dtype=float)
        S1 = fl.Transport(grid, S0, U, dt)
        np.testing.assert_allclose(S1, const*np.ones((N, N), dtype=float))

        # transport constant field with linear velocity
        # add this test -- account for behavior at boundaries

    def test_addForce(self):
        N = 12
        S0 = 3*np.ones((N, N), dtype=float)
        dt = 0.3
        source = 5.6*np.ones((N, N), dtype=float)

        np.testing.assert_allclose(fl.addForce(S0, source, dt),
                                   S0 + dt*source)


if __name__ == '__main__':
    unittest.main()
