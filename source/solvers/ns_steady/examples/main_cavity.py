'''
Example execution script for ns_steady simulations
@author:    David Nolte, dnolte@dim.uchile.cl
@date:      13/09/2016
'''

from dolfin import *
from solvers.ns_steady.nssolver import NSSolver, ReynoldsContinuation
from solvers.ns_steady.nsproblem import NSProblem

# enable optimization (speed up ~ x2)
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = \
    "-O3 -ffast-math -march=native"

# create instance of Navier-Stokes problem with setup from input file
pb = NSProblem('input/cavity.yaml')
# initialize problem
pb.init()

# create solver
sol = NSSolver(pb)
sol.solve()

u, p = sol.w.split(deepcopy=True)
