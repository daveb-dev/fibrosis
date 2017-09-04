from ...nsproblem import NSProblem
from ...nssolver import NSSolver
from dolfin import norm
import pytest


@pytest.fixture
def solve_nitsche():
    pb = NSProblem('cavity_nitsche.yaml')
    pb.options['nonlinear']['method'] = 'newton'
    pb.init()
    sol_nitsche = NSSolver(pb)
    sol_nitsche.solve()

    return sol_nitsche


@pytest.fixture
def solve_strong():
    pb2 = NSProblem('cavity.yaml')
    pb2.options['nonlinear']['method'] = 'newton'
    pb2.init()
    sol_strong = NSSolver(pb2)
    sol_strong.solve()

    return sol_strong


def test_compare_num_iter_nitsche_strong(solve_strong, solve_nitsche):
    ''' Compare strong vs Nitsche no-slip BCs for the BFS test case with Re =
    200 and a newton solver:
        number of iterations
    '''
    assert len(solve_strong.residual) == len(solve_nitsche.residual)


def test_compare_solution_nitsche_strong(solve_strong, solve_nitsche):
    ''' Compare strong vs Nitsche no-slip BCs for the BFS test case with Re =
    200 and a newton solver:
        solution norm
    '''
    w_nitsche = solve_nitsche.w
    (u1, p1) = w_nitsche.split(deepcopy=True)
    w_strong = solve_strong.w
    (u2, p2) = w_strong.split(deepcopy=True)

    print(norm(u1.vector() - u2.vector(), 'l2')/norm(u2.vector(), 'l2'))
    print(norm(p1.vector() - p2.vector(), 'l2')/norm(p2.vector(), 'l2'))

    tol = 1e-13

    # TODO: don't worry, this (probably) was never true. see nse_benchmark.pdf
    # for actual error values (~ 1e4 for 0.025 mesh)

    assert norm(w_nitsche.vector() - w_strong.vector(), 'l2') < tol
