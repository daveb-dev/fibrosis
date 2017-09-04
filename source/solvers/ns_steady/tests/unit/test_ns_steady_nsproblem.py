from ...nsproblem import NSProblem
import numpy as np
import pytest
import yaml
import tempfile
import dolfin
import mshr


@pytest.fixture
def problem():
    tmpfile = tempfile.TemporaryFile()
    tmpfile.write(b'elements: \'Mini\'')
    tmpfile.seek(0)
    pb = NSProblem()
    pb.options = yaml.load(tmpfile)
    assert (type(pb.options) is dict and 'elements' in pb.options and
            pb.options['elements'] == 'Mini')

    return pb


def test_mixed_functionspace(problem):
    problem.mesh = dolfin.UnitSquareMesh(2, 2)
    problem.ndim = 2
    problem.mixed_functionspace()
    assert type(problem.W) is dolfin.functions.functionspace.MixedFunctionSpace


def test_is_enriched(problem):
    problem.mesh = dolfin.UnitSquareMesh(2, 2)
    problem.ndim = 2
    problem.mixed_functionspace()
    assert problem.is_enriched(problem.W)
    assert problem.is_enriched(problem.W.sub(0))
    assert problem.is_enriched(problem.W.sub(0).sub(0))


def test_is_Expression(problem):
    ex = dolfin.Expression(('0.0', '0.0'))
    const = dolfin.Constant('100')
    assert problem.is_Expression(ex)
    assert not problem.is_Expression(const)


def test_is_Constant(problem):
    ex = dolfin.Expression(('0.0', '0.0'))
    const = dolfin.Constant('100')
    assert not problem.is_Constant(ex)
    assert problem.is_Constant(const)


def test_get_boundary_orientation(problem):
    problem.mesh = dolfin.UnitSquareMesh(4, 4)
    problem.bnds = dolfin.MeshFunction('size_t', problem.mesh, 1)
    problem.ndim = 2

    class Left(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < dolfin.DOLFIN_EPS

    class Right(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x[0] > (1.0 - dolfin.DOLFIN_EPS)

    class Top(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x[1] > (1.0 - dolfin.DOLFIN_EPS)

    left = Left()
    right = Right()
    top = Top()

    problem.bnds.set_all(0)
    left.mark(problem.bnds, 1)
    right.mark(problem.bnds, 2)
    top.mark(problem.bnds, 3)

    assert problem._get_boundary_orientation(1) == 0
    assert problem._get_boundary_orientation(2) == 0
    assert problem._get_boundary_orientation(3) == 1


def test_get_inlet_parabola_coef(problem):
    domain = mshr.Rectangle(dolfin.Point(0., 1.), dolfin.Point(5., 3.))
    problem.mesh = mshr.generate_mesh(domain, 4)
    problem.bnds = dolfin.MeshFunction('size_t', problem.mesh, 1)
    problem.ndim = 2

    class Left(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < dolfin.DOLFIN_EPS

    class Top(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x[1] > (3.0 - dolfin.DOLFIN_EPS)

    left = Left()
    top = Top()

    problem.bnds.set_all(0)
    left.mark(problem.bnds, 1)
    top.mark(problem.bnds, 3)

    np.testing.assert_almost_equal(
        problem._get_inlet_parabola_coef(1, 0), np.array([2., 1.]))
    np.testing.assert_almost_equal(
        problem._get_inlet_parabola_coef(3, 1), np.array([2.5, 2.5]))

    np.testing.assert_almost_equal(
        problem._get_inlet_parabola_coef(3, 1, symmetric=True),
        np.array([0., 5.]))
