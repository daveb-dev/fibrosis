''' Solve Stokes equation for a coarted vessel
    see the documentation for details
    
    Parallel: mpirun -np 4 python stokes_func.py
'''

from dolfin import *
import time
import subprocess

if __name__ == "__main__":
    start = time.time()
    from functions import inout as io
    from functions import utils
    from solvers.stokes import compute
    import time
    
    compute()
    
    end	= time.time(); time_solve   = end - start; print "Total time elapsed: %g [seg]" % time_solve
    