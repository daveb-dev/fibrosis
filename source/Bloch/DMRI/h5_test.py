from dolfin import Mesh
from mpi4py import MPI
import h5py

mesh = Mesh()
comm = mesh.mpi_comm()
# comm = MPI.COMM_WORLD
rank = comm.Get_rank()

b = 1
S_TE = 1
with h5py.File("h5_test.h5", "w", comm=comm) as h5:
    h5.create_dataset('b', data=b)
    h5.create_dataset('S', data=S_TE)
    h5.close()

print("process %d" % rank)
