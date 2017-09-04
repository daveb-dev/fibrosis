from ...nsproblem import NSProblem
from ...nssolver import NSSolver

''' The tests snes, newton, qnewton, picard, aitken_converge_to_tol should take
9.5s together '''


def test_snes_converge_to_tol():
    ''' test if snes solver reach tolerance within predetermined number of
    iterations (REGRESSION TEST) for Re = 1000 '''

    pb = NSProblem('cavity.yaml')
    assert pb.options['nonlinear']['method'] == 'snes', (
        'test not configured correctly! expected SNES.')
    pb.init()
    snes_sol = NSSolver(pb)
    snes_sol.solve()

    assert len(snes_sol.residual) <= 11


def test_newton_converge_to_tol():
    ''' test if manual newton solver reach tolerance within predetermined
    number of iterations (REGRESSION TEST) for Re = 1000 '''

    pb = NSProblem('cavity.yaml')
    pb.options['nonlinear']['method'] = 'newton'
    pb.init()
    newton_sol = NSSolver(pb)
    newton_sol.solve()

    assert len(newton_sol.residual) <= 11


def test_qnewton_converge_to_tol():
    ''' test if quasi-newton solver reach tolerance within predetermined
    number of iterations (REGRESSION TEST) for Re = 1000 '''
    pb = NSProblem('cavity.yaml')
    pb.options['nonlinear']['method'] = 'qnewton'
    pb.init()
    qnewton_sol = NSSolver(pb)
    qnewton_sol.solve()

    assert len(qnewton_sol.residual) <= 11


def test_picard_converge_to_tol():
    ''' test if picard solver reach tolerance within predetermined
    number of iterations (REGRESSION TEST) for Re = 1000 '''
    pb = NSProblem('cavity.yaml')
    pb.options['nonlinear']['method'] = 'picard'
    pb.init()
    picard_sol = NSSolver(pb)
    picard_sol.solve()

    assert len(picard_sol.residual) <= 51


def test_aitken_converge_to_tol():
    ''' test if Picard + Aitken solves Re = 1000 case within 40 iterations '''
    pb = NSProblem('cavity.yaml')
    pb.options['nonlinear']['method'] = 'picard'
    pb.options['nonlinear']['use_aitken'] = True
    pb.init()
    picard_sol = NSSolver(pb)
    picard_sol.solve()

    assert len(picard_sol.residual) <= 42
