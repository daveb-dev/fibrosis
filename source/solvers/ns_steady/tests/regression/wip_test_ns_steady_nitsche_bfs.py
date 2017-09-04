from ...nsproblem import NSProblem
from ...nssolver import NSSolver
from dolfin import norm
import pytest


@pytest.fixture
def solve_nitsche():
    pb = NSProblem('bfs_nitsche.yaml')
    pb.options['nonlinear']['method'] = 'newton'
    pb.init()
    sol_nitsche = NSSolver(pb)
    sol_nitsche.solve()

    return sol_nitsche


@pytest.fixture
def solve_strong():
    pb2 = NSProblem('bfs.yaml')
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
    w_strong = solve_strong.w

    tol = 1e-13

    assert norm(w_nitsche.vector() - w_strong.vector(), 'l2') < tol
