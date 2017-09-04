''' utils module
    for functions of general use and utility'''
import shutil


def on_cluster():
    import platform
    import re
    uname = platform.uname()
    pattern = re.compile("^(leftraru\d)|(cn\d\d\d)")
    return bool(pattern.match(uname[1]))


def prep_mesh(mesh_file):
    if on_cluster():
        # running on NLHPC cluster
        # TODO: UNTESTED CHANGE. Test this.
        # mfile = '/home/dnolte/fenics/nitsche/meshes/' + mesh_file
        mfile = shutil.os.getcwd() + '/' + mesh_file
        mesh = '/dev/shm/' + mesh_file
        shutil.copy(mfile, mesh)
    else:
        # running on local machine
        mesh = mesh_file

    return mesh


def trymkdir(path):
    ''' mkdir if path does not exist yet '''
    import os
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

