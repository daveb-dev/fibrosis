''' Solve Stokes equation for a coarted vessel
    see the documentation for details
    
    Parallel: mpirun -np 4 python stokes_func.py
'''

from dolfin import *
import time

if __name__ == "__main__":
    start = time.time()
    from functions import inout as io
    from functions import utils
    from solvers.stokes_transient import compute

    compute()
    
    end	= time.time(); time_solve   = end - start; 
    if rank == 0:
      print "Total time elapsed: %g [seg]" % time_solve
    