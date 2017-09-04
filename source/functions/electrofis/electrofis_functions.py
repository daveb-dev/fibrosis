# Utilities for electrophisiology problem
  
import os.path
def FibroMesh_all(prms):
    
    a		    = prms['mesh']['a']
    b		    = prms['mesh']['b']
    base	    = prms['mesh']['base']
    altura	    = prms['mesh']['altura']
    is_random	= prms['mesh']['random']
    if not is_random:
        theta_c	    = prms['mesh']['theta_c']
        theta_f	    = prms['mesh']['theta_f']

    mesh_h_file = './meshes/homogenized_' + str(int(base)) + "x" + str(int(altura)) + '.h5'
    if is_random:
        mesh_file  = './meshes/random_fibro_'    + str(int(base)) + "x" + str(int(altura)) + "_" + str(int(a*10)) + "_" + str(int(b*10)) + '.h5'
        theta_file = './meshes/theta_functions_' + str(int(base)) + "x" + str(int(altura)) + "_" + str(int(a*10)) + "_" + str(int(b*10)) + '.h5'
    else:
        mesh_file  = './meshes/fibro_'    + str(int(base)) + "x" + str(int(altura)) + "_" + str(int(a*10)) + "_" + str(int(b*10)) + '_' + str(int(theta_c*100)) + '_' + str(int(theta_f*100)) + '.h5'
        theta_file = './meshes/theta_functions_' + str(int(base)) + "x" + str(int(altura)) + "_" + str(int(a*10)) + "_" + str(int(b*10)) + '.h5'
    if not os.path.isfile(mesh_file) or not os.path.isfile(mesh_h_file) or not os.path.isfile(theta_file): 
        from functions.fibro_mesh_random import generate_mesh
        generate_mesh(prms)
    return mesh_file, mesh_h_file, theta_file
