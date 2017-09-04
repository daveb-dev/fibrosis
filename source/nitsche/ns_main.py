from dolfin import *
from solvers.ns_steady.nssolver import NSSolver, ReynoldsContinuation
from solvers.ns_steady.nsproblem import NSProblem

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = \
    "-O3 -ffast-math -march=native"

# pb = NSProblem('input/cavity.yaml')
# pb = NSProblem('input/bfs.yaml')
#  pb = NSProblem('input/coarc.yaml')
# pb = NSProblem('input/coarc_ref.yaml')
# pb = NSProblem('input/coarc_estim.yaml')
# pb = NSProblem('input/pipe_estim.yaml')
pb = NSProblem('experiments/navierslip_transpiration_test/input/s1c_ref.yaml')
# pb = NSProblem('input/coarc_valid_estim.yaml')
#  pb = NSProblem('input/pipe.yaml')

pb.init()
sol = NSSolver(pb)
sol.solve()

# char_len = 1.
# ubulk = 2./3.*20
# rec = ReynoldsContinuation(pb, char_len, ubulk, Re_start=10, Re_num=10,
#                            Re_end=800, init_start=2, init_end=5)
# rec.solve()


def save_pvd(sol, fileprefix, path='./'):
    ''' Extract u, p fields from solver object and save to
    path/fileprefix_u/p.pvd files '''
    pfile = path + '/' + fileprefix + '_p.pvd'
    ufile = path + '/' + fileprefix + '_u.pvd'
    u, p = sol.w.split(deepcopy=True)
    u.rename('u', 'velocity')
    p.rename('p', 'pressure')
    f1 = File(ufile)
    f1 << u
    f1 = File(pfile)
    f1 << p
    pass


def save_xdmf(sol, fileprefix, path='./'):
    ''' Extract u, p fields from solver object and save to
    path/fileprefix_u/p.xdmf files '''
    pfile = path + '/' + fileprefix + '_p.xdmf'
    ufile = path + '/' + fileprefix + '_u.xdmf'
    u, p = sol.w.split(deepcopy=True)
    u.rename('u', 'velocity')
    p.rename('p', 'pressure')
    XDMFFile(ufile).write(u)
    XDMFFile(pfile).write(p)
    pass
