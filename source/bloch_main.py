''' Sleek variant of the DMRI FEM simulator '''
from solvers.bloch_simple import *
from functions import inout

import sys

from dolfin import *
from mpi4py import MPI

set_log_level(30)

# compiler optimization
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"
# intel compiler (cluster)
# parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -xHost -ipo"  #
# gcc
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math \
                                                     -march=native"

# prevent output from rank > 0 processes
if MPI.COMM_WORLD.rank > 0:
    sys.stdout = open("/dev/null", "w")
    # and handle C++ as well
    from ctypes import *
    libc = CDLL("libc.so.6")
    stdout = libc.fdopen(1, "w")
    libc.freopen("/dev/null", "w", stdout)

if __name__ == '__main__':

    print("rank: %i" % MPI.COMM_WORLD.Get_rank())
    # param = inout.prms_load('/home/dnolte/fenics/input/prms_bloch.yaml')
    param = inout.prms_load('./input/prms_bloch.yaml')

    T = sum(param['phys']['pseq']['dt'])
    T = 0.01
    dt = param['num']['dt']
    bval = param['phys']['b']
    if not type(bval) == list:
        bval = [bval]

    signal_b = []
    for b in bval:
        t = 0.
        meshdata, ppdata, solver = initialize(param, b)
        u, forms = variational_problem(param, meshdata)
        postproc(param, ppdata, meshdata, u, t, T)  # ouput initial condition
        mat = preassemble(param, forms)

        while t <= T:
            t += dt
            A, b = assemble_LS(param, mat, t, u)
            u = timestep(solver, A, b, u)
            sig, ppdata = postproc(param, ppdata, meshdata, u, t, T)
        signal_b.append(sig)

    # Need an idea for this.
    # If the time loop is not inside the solver module, and less the loop
    # over b values, postproc() cannot handle the signal(b) output
    if MPI.COMM_WORLD.Get_rank() == 0:
        import h5py
        fname = "signal_b_TE.h5"
        fdir = param['io']['results']
        with h5py.File(fdir+"/"+fname, "w") as h5:
            h5.create_dataset('b', data=bval)
            h5.create_dataset('S', data=signal_b)
            h5.close()
#
