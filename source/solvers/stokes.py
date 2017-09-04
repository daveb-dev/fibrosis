''' solver module '''
from dolfin import *

def compute():
    ''' Stokes stationary flow equation solver
        author: Felipe Galarce
        email: felipe.galarce.m@gmail.com
    '''

    from functions.inout import readmesh
        
    sol   = 'mumps'                       # Krylov solver settings
    prec  = 'none'                        # Preconditioner (use 'none' to not use preconditioner)
    mu    = 0.035                         # dynamic viscocity
    g_inx = 1                             # Neumann x component of boundary condition over gamma_inlet
    h     = 0.15
    ST_fspace_order = 'P1bP1'
    #parameters["form_compiler"]["quadrature_degree"] = 6

    
    mesh_file   = './meshes/COARTED_60_h' + str(h) + '.h5'    
    #results_dir = './brinkman_h' + str(h) + '.h5'
    results_dir = './results/test_functions/stokes_h' + str(h) + '_'+ ST_fspace_order
    
    
    # Read mesh, boundaries and subdomains
    mesh, subdomains, boundaries = readmesh(mesh_file)

    # Define function spaces
    if ST_fspace_order == 'P1bP1':
      P1 = VectorFunctionSpace(mesh, "Lagrange", 1)
      B  = VectorFunctionSpace(mesh, "Bubble", 4)
      V  = P1 + B
      Q  = FunctionSpace(mesh, "CG",  1)
    
    if ST_fspace_order == 'P2P1':
      V = VectorFunctionSpace(mesh, "Lagrange", 2)
      Q = FunctionSpace(mesh, "CG",  1)
    
    if ST_fspace_order == 'P3P2':
      V = VectorFunctionSpace(mesh, "Lagrange", 3)
      Q = FunctionSpace(mesh, "CG",  2)    
    
    Mini = V*Q

    # Mark UFL variable with boundaries
    ds   = Measure('ds', domain=mesh  , subdomain_data=boundaries)

    # Define variational problem
    (u, p) = TrialFunctions(Mini)
    (v, q) = TestFunctions(Mini)
    f = Constant((0, 0, 0))
    a = (mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u))*dx # note the change in the continuity sign in order to get a symmetric matrix
    noslip = project(Constant((0, 0, 0)), V)
    bc = DirichletBC(Mini.sub(0), noslip, boundaries, 2)
    L = inner(f, v)*dx  + inner(Constant((0,g_inx,0)), v)*ds(1)           

    if has_petsc():
      PETScOptions.set("mat_mumps_icntl_14", 80.0)

    # Compute solution
    w = Function(Mini)
    rank = mesh.mpi_comm().Get_rank()
    solve(a == L, w, bc, solver_parameters={'linear_solver': sol, 'preconditioner': prec})    

    # Split the mixed solution using a shallow copy
    (u, p) = w.split()

    plot(u);  plot(p); interactive()

    # Save solution to file (to be read as weighting function in pressure drop estimation)
    u_file = HDF5File(mesh.mpi_comm(), results_dir + "/velocity.h5", "w")
    p_file = HDF5File(mesh.mpi_comm(), results_dir + "/pressure.h5", "w")
    u_file.write(u, "velocity")
    p_file.write(p, "pressure")
    u_file.close(); u_file.close();

    # Save solution to file (to be read in paraview)
    ufile_pvd = File(results_dir + "/velocity_paraview.xdmf");  ufile_pvd << u
    pfile_pvd = File(results_dir + "/pressure_paraview.xdmf");  pfile_pvd << p
