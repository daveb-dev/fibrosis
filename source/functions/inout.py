''' input/output module '''


def read_HDF5_data(h5file, mpi_comm, fun, name):
    ''' Read checkpoint data from a HDF5 file into a dolfin function.

    Args:
        h5file (str)    HDF5 File to be read from
        mpi_comm        MPI comm, e.g. mesh.mpi_comm()
        fun             Dolfin function
        name (str)      name of the hdf5 dataset
    '''
    from dolfin import HDF5File
    hdf = HDF5File(mpi_comm, h5file, "r")
    hdf.read(fun, name)
    hdf.close()
    pass


def write_HDF5_data(h5file, mpi_comm, fun, name):
    ''' Write checkpoint data from a dolfin function into a HDF5 file for
    reuse.

    Args:
        h5file (str)    HDF5 File to be read from
        mpi_comm        MPI comm, e.g. mesh.mpi_comm()
        fun             Dolfin function
        name (str)      name of the hdf5 dataset
    '''
    from dolfin import HDF5File
    hdf = HDF5File(mpi_comm, h5file, "w")
    hdf.write(fun, name)
    hdf.close()
    pass


def read_mesh(mesh_file):
    ''' Read HDF5 or DOLFIN XML mesh.

    Args:
        mesh_file       path to mesh file

    Returns:
        mesh            Mesh
        sd              subdomains
        bnd             boundaries
    '''
    # TODO: exceptions, files exist?
    from dolfin import Mesh, MeshFunction, CellFunction, HDF5File, \
        FacetFunction
    # pth = '/'.join(mesh_file.split('/')[0:-1])
    tmp = mesh_file.split('.')  # [-1].split('.')
    mesh_type = tmp[-1]
    mesh_pref = '.'.join(tmp[0:-1])

    if mesh_type == 'xml':
        mesh = Mesh(mesh_file)
#        rank = mesh.mpi_comm().Get_rank()
        rank = 0
        try:
            subdomains = MeshFunction("size_t", mesh,
                                      mesh_pref+"_physical_region.xml")
        except:
            # if rank == 0:
            #     print('no subdomain file found (%s)' %
            #           (mesh_pref+"_physical_region.xml"))
            subdomains = CellFunction("size_t", mesh)
        try:
            boundaries = MeshFunction("size_t", mesh,
                                      mesh_pref+"_facet_region.xml")
        except:
            if rank == 0:
                print('no boundary file found (%s)' %
                      (mesh_pref+"_facet_region.xml"))
            boundaries = FacetFunction("size_t", mesh)

    elif mesh_type == 'h5':
        mesh = Mesh()
#        rank = mesh.mpi_comm().Get_rank()
        rank = 0

        hdf = HDF5File(mesh.mpi_comm(), mesh_file, "r")
        hdf.read(mesh, "/mesh", False)
        subdomains = CellFunction("size_t", mesh)
        boundaries = FacetFunction("size_t", mesh)
        if hdf.has_dataset('subdomains'):
            hdf.read(subdomains, "/subdomains")
        # else:
        #     if rank == 0:
        #         print('no <subdomains> datasets found in file %s' %
        #               mesh_file)
        if hdf.has_dataset('boundaries'):
            hdf.read(boundaries, "/boundaries")
        else:
            if rank == 0:
                print('no <boundaries> datasets found in file %s' %
                      mesh_file)

    elif mesh_type in ['xdmf', 'xmf']:
        import sys
        sys.exit('XDMF not supported yet. Use HDF5 instead!')
    else:
        import sys
        sys.exit('mesh format not recognized. try XML (serial) or HDF5')

# NOTE http://fenicsproject.org/qa/5337/importing-marked-mesh-for-parallel-use
    # see file xml2xdmf.py
    return mesh, subdomains, boundaries


def readmesh(mesh_file):
    ''' Deprecated function. Use read_mesh instead.'''
    raise Exception('readmesh was renamed to read_mesh (PEP8 conforming).')


def prms_load(infile):
    ''' Deprecated function. Use read_parameters instead.'''
    raise Exception('prms_load was renamed to read_parameters (PEP8 '
                    'conforming).')


def read_parameters(infile):
    ''' Read in parameters yaml file.

    Args:
        infile      path to yaml file

    Return:
        prms        parameters dictionary
    '''
    import yaml
    try:
        with open(infile, 'r+') as f:
            prms = yaml.load(f)
            f.close()
    except IOError:
        import sys
        sys.exit('error: file not found: %s' % infile)
    return prms


def prms_print(prms):
    ''' Deprecated function. Use print_parameters instead.'''
    raise Exception('prms_print was renamed to print_parameters (PEP8 '
                    'conforming).')


def print_parameters(prms):
    ''' Print parameter dictionary in human readable form.

    Args:
        prms        parameters dictionary
    '''
    import yaml
    print(yaml.dump(prms))
    pass
