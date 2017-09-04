''' Solve Navier-Stokes equation
    see the documentation for details
    
    Parallel: mpirun -np 4 python stokes_func.py
'''

if __name__ == "__main__":

    from functions import inout as io
    from functions import utils

    # choose solver to be used (options: monolitic, monolitic_direct, chorin, LU, yosida, cristobal)
    solver = 'monolitic_direct'

    if solver == 'monolitic': from solvers.navier_stokes_monolitic import compute
    if solver == 'monolitic_direct': from solvers.navier_stokes_monolitic_DIRECT import compute
    if solver == 'chorin': from solvers.navier_stokes_chorin_teman import compute
    if solver == 'LU': from solvers.navier_stokes_LU import compute
    if solver == 'yosida': from solvers.navier_stokes_yosida import compute
    if solver == 'cristobal': from solvers.navier_stokes_cristobal import compute
    compute()