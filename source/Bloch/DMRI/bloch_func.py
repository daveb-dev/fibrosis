''' Solve Bloch equation for 1 voxel
    Approach: define traverse magnetization vector m = (mx, my)
      (considering complex magnetization: mx is the real part and my the
    imaginary part of the complex magnetization)
    and solve coupled 2D vector Bloch Equation in a 3D voxel,
    using a MixedFunctionSpace VxV

    Status: solve Bloch Eq. with
        - gradient constant in time & space
        - 2 subdomains with different diffusion coefficients
        - using implicite backward diff
        - MixedFunctionSpace

    Units: x: mm, D: mm^2/s

    Parallel: mpirun -np 4 python bloch.py

    Parallel requires HDF5 mesh. Convert MSH to XML with dolfin-converter,
    then XML to HDF5 with script xml2hdf5.py -- better procedure needed!
    Watch github.com/nschloe/meshio
'''
#   TODO: VERIFICATION -- ANALYTICAL SOLUTIONS
#   DONE: realistic physical quantities
#       NOTE: adimensionalize??
#   DONE: realistic gradients (strength, waveform, timing)
#   DONE: check stability for explicit solver (just use implicite)
#   TODO: periodic boundaries (s. Xu (2007), Russel (2012), Nguyen (2014))
#       NOTE: FEniCS not supporting PBs for DG?
#   TODO: implement flow (1. plug flow, 2. Poiseuille(?), 3. Stokes) and
#         stabilize (s. Advection-diffusion & Navier-Stokes demos)
#   ???: compute signal (simulate k-space)
#   TODO: interface jump condition (permeability, Robin BCs --> DG)
#   DONE: go parallel (mpi) -> TODO next: go cluster
#   DONE: pre-assemble matrices

#   XXX: N antisymetric -> GMRES?

#   TODO: need more efficient mesh conversion procedure for GMSH -> HDF5

#   XXX: SIGNAL PRACTICALLY INSENSITIVE TO DIFFUSION COEFFICIENT
#           ADAPT GRADIENT (b-value), TIMES ETC??
#   XXX: SOMETIMES HDF5 TIME SERIES OUPUT DOESNT WORK (mpi?)

from dolfin import *
# import numpy as np
from mpi4py import MPI
import h5py
# import os
# import sys
# import inspect


set_log_level(30)  # WARNING


if __name__ == "__main__":
    # add path to functions module
    # cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(
    #     os.path.split(inspect.getfile(inspect.currentframe()))[0],
    #     "../functions")))
    # if cmd_subfolder not in sys.path:
    #     sys.path.insert(0, cmd_subfolder)

    # sys.path.append('../..')

    # path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # if not path in sys.path:
    #     sys.path.insert(1, path)
    # del path

    from functions import inout as io
    from functions import utils
    from solvers.bloch_func import compute

    # Load simulation parameters from yaml file
    parameter_file = './input/prms_bloch.yaml'
    prms = io.prms_load(parameter_file)

    # Get MPI comm
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Make results directory if applicable
    utils.trymkdir(prms['io']['results'])

    # Print parameters on 1st proc only
    if rank == 0:
        io.prms_print(prms)

    # Extract b values and make list if int/float
    b = prms['phys']['b']
    if not type(b) == list:
        b = [b]

    # Loop over all b values
    S_TE = []
    for bval in b:
        if rank == 0:
            print("b = %i" % bval)

        # Compute complex signal at time 'TE'
        res = compute(bval, prms)
        S_TE.append(res)

    if prms['io']['signal_TE'] == 1:
        with h5py.File(prms['io']['results']+"/signal_b_TE.h5", "w",
                       comm=comm) as h5:
            h5.create_dataset('b', data=b)
            h5.create_dataset('S', data=S_TE)
            h5.close()
