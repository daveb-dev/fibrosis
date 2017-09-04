def generate_mesh(prms):
    # Generate fibrotic mesh using GMSH
    # The labels are:
    #   collagen subdomains: 5
    #   healthy tissue: 6
    #   left boundary: 4
    #   right boundary: 2
    #   top boundary: 3
    #   bottom boundary: 1
    #
    #	Felipe Galarce Marin - 2016
    #	felipe.galarce.m@gmail.com

    from dolfin import *
    import pygmsh as pg
    import numpy as np
    import utils
    import subprocess

    # Extract parameters
    a 	    = float(prms['mesh']['a'])
    b 	    = float(prms['mesh']['b'])
    base    = float(prms['mesh']['base'])
    altura  = float(prms['mesh']['altura'])
    desp    = float(prms['mesh']['desp'])
    is_random =     prms['mesh']['random']   
    if not is_random:
        theta_c_local, theta_f_local =  prms['mesh']['theta_c'],  prms['mesh']['theta_f']

    l_coarse = 0.5
    l_unrefine		= 1.5 # mesh size in coarser parts of the mesh
    geom = pg.Geometry()

    X = [[0.0, 0.0, 0.0], [base + 2*b*desp, 0.0, 0.0], [base + 2*b*desp, altura + 2*a*desp, 0.0], [0.0, altura + 2*a*desp, 0.0]];
    surface_id = geom.add_polygon(X, l_coarse/5, holes=None)

    # save array with local values of theta in order to create theta functions
    theta_c_array = np.empty((int(np.floor(base/b))*int(np.floor(altura/a)))); kkk = 0
    theta_f_array = np.empty((int(np.floor(base/b))*int(np.floor(altura/a)))); 

    # parameters of the normal distribution TODO: poner en prms file
    theta_f_mean = 0.5
    theta_c_mean = 0.3
    sigma_c      = 0.2
    sigma_f      = 0.3
    # ======== CREATE MESH FOR EXACT PROBLEM =========
    # create collagen subdomains
    collagen_subdomain, k = '{', 0; 
    desp_x = 0; # corre cada fila por medio de colageno un poco a la derecha
    for Px in range(0, int(np.floor(base/b))):
        for Py in range(0, int(np.floor(altura/a))):
            if is_random:
                # randomly generate fraction of fibrotic tissue, in realistic ranges
                theta_c_local, theta_f_local = 2, 2
                while (theta_c_local > 0.4 or theta_c_local < 0.15):
                  theta_c_local = abs(np.random.normal(theta_c_mean, sigma_c))
                while (theta_f_local < 0.35 or theta_f_local > 0.9):
                  theta_f_local = abs(np.random.normal(theta_f_mean, sigma_f))


            Pxx, Pyy = float(Px*b - b*theta_f_local/2 + b*desp + desp_x*b/2), float(Py*a - a*theta_c_local/2 + a*desp)
            x1 = [Pxx,        		        Pyy, 				        0.0];
            x2 = [Pxx + b*theta_f_local, 	Pyy, 				        0.0];
            x3 = [Pxx + b*theta_f_local,	Pyy + a*theta_c_local, 		0.0];
            x4 = [Pxx,        		        Pyy + a*theta_c_local, 		0.0];

            l_size = (theta_c_local < theta_f_local)*theta_c_local*a + (theta_c_local >= theta_f_local)*theta_f_local*b;

            x1_label = geom.add_point_in_surface(x1, l_size, surface_id);
            x2_label = geom.add_point_in_surface(x2, l_size, surface_id);
            x3_label = geom.add_point_in_surface(x3, l_size, surface_id);
            x4_label = geom.add_point_in_surface(x4, l_size, surface_id);

            l1_label = geom.add_line(x1_label, x2_label);
            l2_label = geom.add_line(x2_label, x3_label);
            l3_label = geom.add_line(x3_label, x4_label);
            l4_label = geom.add_line(x4_label, x1_label);    
            llp      = geom.add_line_loop((l1_label, l2_label, l3_label, l4_label))
            collagen = geom.add_plane_surface(llp)

            if k == 0:
              collagen_subdomain = collagen_subdomain + collagen
              k = 1
            else:
              collagen_subdomain = collagen_subdomain +  ", " + collagen

            l1_label = geom.add_line_in_surface(l1_label, surface_id);
            l2_label = geom.add_line_in_surface(l2_label, surface_id);
            l3_label = geom.add_line_in_surface(l3_label, surface_id);
            l4_label = geom.add_line_in_surface(l4_label, surface_id);

            theta_f_array[kkk], theta_c_array[kkk] = theta_f_local, theta_c_local; kkk = kkk + 1

            if is_random:
                if desp_x == 1:
                    desp_x = 0
                else:
                    desp_x = 1

    collagen_subdomain = collagen_subdomain + '}'

    # unrefine place where fine mesh is not required
    unrefine_points 	= 0.2*np.array(range(1, int(altura*5)))
    xx = desp - b*0.5 + b*np.array(range(1, int(base/b)))

    if not is_random:
        for Py in unrefine_points:
            for Px in xx:
                geom.add_point_in_surface([Px, Py, 0], l_unrefine, surface_id)

    geom.set_physical_objects(collagen_subdomain)
    
    # Save mesh and convert to .h5 format
    if is_random:
        out_name = 'random_fibro_' + str(int(base)) + "x" + str(int(altura)) + "_" + str(int(a*10)) + "_" + str(int(b*10))
    else:
        out_name = 'fibro_'    + str(int(base)) + "x" + str(int(altura)) + "_" + str(int(a*10)) + "_" + str(int(b*10)) + '_' + str(int(theta_c_local*100)) + '_' + str(int(theta_f_local*100))
    utils.trymkdir(prms['io']['results'])
    FILE = open("./fibro_file.geo", 'w+')
    FILE.write(geom.get_code()); FILE.close();
    subprocess.call("cp ./functions/geo2h5.sh ./"                                       , shell=True)
    subprocess.call("cp ./functions/xml2hdf5.py ./"                                     , shell=True)
    subprocess.call("./geo2h5.sh" + " fibro_file " + out_name                           , shell=True)
    subprocess.call("cp ./" + out_name + ".h5 ./meshes/"                                , shell=True)
    subprocess.call("rm ./geo2h5.sh ./xml2hdf5.py " + out_name + ".h5 fibro_file.geo"   , shell=True)

    # ======== CREATE MESH FOR HOMOGENIZED PROBLEM =========    
    geom = pg.Geometry()
    X = [[0.0, 0.0, 0.0], [base + 2*b*desp, 0.0, 0.0], [base + 2*b*desp, altura + 2*a*desp, 0.0], [0.0, altura + 2*a*desp, 0.0]];
    surface_id = geom.add_polygon(X, l_coarse, holes=None)
    geom.set_physical_objects_homo()
    # Save mesh and convert to .h5 format
    out_name = 'homogenized_' + str(int(base)) + "x" + str(int(altura))
    FILE = open("./fibro_file.geo", 'w+')
    FILE.write(geom.get_code()); FILE.close();
    subprocess.call("cp ./functions/geo2h5.sh ./"                                       , shell=True)
    subprocess.call("cp ./functions/xml2hdf5.py ./"                                     , shell=True)
    subprocess.call("./geo2h5.sh" + " fibro_file " + out_name                           , shell=True)
    subprocess.call("cp ./" + out_name + ".h5 ./meshes/"                                , shell=True)
    subprocess.call("rm ./geo2h5.sh ./xml2hdf5.py " + out_name + ".h5 fibro_file.geo"   , shell=True)

    # ======== CREATE THETA FUNCTIONS OVER HOMOGENIZED MESH =========
    from functions.inout import readmesh
    mesh, subdomains, boundaries = readmesh('./meshes/' + out_name + '.h5')

    V = FunctionSpace(mesh, 'CG', 1)
    theta_c_function, theta_f_function = Function(V), Function(V); kkk = 0; 

    dofmap = V.dofmap(); dofs = dofmap.dofs()

#    THE FOLLOWING CODE IS DEPRECATED BEFORE 1.6 VERSION
#    # mesh dimension and dofmap
#    gdim = mesh.geometry().dim()
#    #Get coordinates of the dofs
#    dofs_coor = dofmap.tabulate_all_coordinates(mesh).reshape((-1, gdim)) DEPRECATED AFTER 1.6 VERSION
#    for Px in range(0, int(np.floor(base/b))):
#        for Py in range(0, int(np.floor(altura/a))):
#            Pxx, Pyy = float(Px*b + b*desp - b/2), float(Py*a + a*desp - a/2)
#            for dof, dof_coor in zip(dofs, dofs_coor):
#                if (dof_coor[0] >= Pxx and dof_coor[0] < Pxx + b) and (dof_coor[1] >= Pyy and dof_coor[1] < Pyy + a):
#                    theta_c_function.vector()[dof], theta_f_function.vector()[dof] = theta_c_array[kkk], theta_f_array[kkk]
#            kkk = kkk + 1

    #   Get coordinates of the dofs
    dofs_coor   = V.tabulate_dof_coordinates()
    desp_x = 0; # corre cada fila por medio de colageno un poco a la derecha
    for Px in range(0, int(np.floor(base/b))):
        for Py in range(0, int(np.floor(altura/a))):
            Pxx, Pyy = float(Px*b + b*desp - b/2 + desp_x*b/2), float(Py*a + a*desp - a/2)
            sss = 0
            for dof in dofs:
                dof_x, dof_y = dofs_coor[sss], dofs_coor[sss + 1]; sss = sss + 2
                if (dof_x >= Pxx and dof_x < Pxx + b) and (dof_y >= Pyy and dof_y < Pyy + a):
                    theta_c_function.vector()[dof], theta_f_function.vector()[dof] = theta_c_array[kkk], theta_f_array[kkk]
            kkk = kkk + 1

            if is_random:
                if desp_x == 1:
                    desp_x = 0
                else:
                    desp_x = 1

    # save theta functions to file
    out_name = 'theta_functions_' + str(int(base)) + "x" + str(int(altura)) + "_" + str(int(a*10)) + "_" + str(int(b*10)) + '.h5'
    hdf = HDF5File(mesh.mpi_comm(), "./meshes/" + out_name, "w")
    hdf.write(theta_f_function, "/theta_f_function")
    hdf.write(theta_c_function, "/theta_c_function")
    hdf.close()
