'''
Convert XML mesh to HDF5 using DOLFIN
'''
from dolfin import *
import os.path


def _main():
    # Parse command line arguments.
    args = _parse_options()

    fin = args.infile

    tmp = fin.split('.')
    fname = '.'.join(tmp[0:len(tmp)-1])
    ftype = tmp[-1]

    # check if file has the right format
    if not ftype == 'xml':
        print('input file is not in XML format')
        return -1
    if not os.path.isfile(fin):
        print('file not found')
        return -1

    # read mesh
    print('reading DOLFIN mesh '+fin)
    mesh = Mesh(fin)

    # write mesh
    print('write HDF5 mesh '+fname+'.h5')
    hdf = HDF5File(mesh.mpi_comm(), fname+".h5", "w")
    hdf.write(mesh, "/mesh")

    # if files exist, write subdomain and boundary information
    print('look for boundaries and subdomains')
    subdomains = None
    boundaries = None
    if os.path.isfile(fname+"_physical_region.xml"):
        print('write subdomains to '+fname+'.h5')
        subdomains = MeshFunction("size_t", mesh, fname+"_physical_region.xml")
        hdf.write(subdomains, "/subdomains")
    if os.path.isfile(fname+"_facet_region.xml"):
        print('write boundaries to '+fname+'.h5')
        boundaries = MeshFunction("size_t", mesh, fname+"_facet_region.xml")
        hdf.write(boundaries, "/boundaries")

    return


def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser(description='Convert XML to HDF5.')

    parser.add_argument('infile', type=str, help='.xml file to be read from')

    # parser.add_argument(
    #     'outfile',
    #     type=str,
    #     help='.h5 file to be written to'
    #     )

    return parser.parse_args()


if __name__ == '__main__':
    _main()
