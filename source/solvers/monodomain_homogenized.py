""" solver module """
from dolfin import *

def compute(prms):
    ''' Homogenized reaction-diffusion equations solver
        author: Felipe Galarce
        email: felipe.galarce.m@gmail.com
    '''
    from functions.inout import readmesh
    from functions.electrofis.electrofis_functions import FibroMesh_all
    from functions.homogenization import generate_efective_tensor
    import numpy as np
    import time
    import sys
    import subprocess

    # extract parameters
    results_dir     = prms['io']['results']
    dt              = prms['num']['dt']
    T               = prms['num']['T']
    reaction_model  = prms['phys']['reaction_model']

    # configuring form compiler
#    parameters["form_compiler"]["quadrature_degree"] = 3
    parameters["form_compiler"]["optimize"]          = True
    parameters["form_compiler"]["representation"]    = 'quadrature'

    # CLEANING UP RESULTS DIRECTORY
#    subprocess.call("rm -r " + results_dir, shell=True)

    # LOAD MESHES
    mesh_file, mesh_h_file, theta_file = FibroMesh_all(prms)
    mesh, subdomains, boundaries        = readmesh(mesh_file)
    mesh_h, subdomains_h, boundaries_h  = readmesh(mesh_h_file)

    # DECLARE FUNCTION SPACES
    V, Vh, TS = FunctionSpace(mesh, "CG", 1),  FunctionSpace(mesh_h, "CG", 1), TensorFunctionSpace(mesh_h, "CG", 1)
    
    # LOAD THETA FUNCTIONS 
    hdf = HDF5File(mesh.mpi_comm(), theta_file, "r")
    theta_c, theta_f = Function(Vh), Function(Vh)
    hdf.read(theta_c, "/theta_c_function"); hdf.read(theta_f, "/theta_f_function")
    hdf.close()

    # DEFINE MEASURES
    ds, dsh   = Measure('ds', domain=mesh   , subdomain_data=boundaries), Measure('ds', domain=mesh_h , subdomain_data=boundaries_h)

    # mesh dimension and dofmap
    gdim = mesh_h.geometry().dim()
    dofmap = Vh.dofmap(); dofs = dofmap.dofs()

    Deff, Deffp = Function(TS), Function(TS)
    Deff, C = generate_efective_tensor(prms, subdomains, theta_c, theta_f, dofs, Deff, Deffp)

    phi,  v 	= TrialFunction(V), TestFunction(V)
    phi0, phit 	= Function(V), Function(V)
    phih, vh 	= TrialFunction(Vh), TestFunction(Vh)
    phi0h, phith= Function(Vh), Function(Vh)
 
    M,  K  = assemble(inner(phi, v)*dx), assemble(inner(C * grad(phi), grad(v)) * dx)
    Mh, Kh = assemble(inner(phih, vh)*dx), assemble(inner(Deff*grad(phih), grad(vh))*dx)
    A 		= 1/dt*M  + K
    Ah      = 1/dt*Mh + Kh

    # variational form and pre-assembly
    if reaction_model == 1:
        # TODO: load parameters using a function 
        # declare test and trial functions and assemble common matrix outside this
        alpha = 0.08 
        c1    = 0.175
        c2    = .03
        bb     = 0.011
        d     = .55

        r, rh  = Function(V), Function(Vh)
        r0, r0h = Function(V), Function(Vh)
        r_file, rh_file  = File(results_dir + "/r_FHN.pvd"), File(results_dir + "/rh_FHN.pvd")
        A  = (1/dt + c1*alpha)*M + K
        Ah  = (1/dt + c1*alpha)*Mh + Kh
    
    # Configure solver and file to save solution
    solver, solverh = PETScKrylovSolver("cg", "amg"), PETScKrylovSolver("cg", "amg")
    solver.set_reuse_preconditioner(True); solverh.set_reuse_preconditioner(True)
    solver.set_nonzero_guess(True);  solverh.set_nonzero_guess(True)

    phi_file, phih_file = File(results_dir + "/" + str(reaction_model) + "/potential.pvd"), File(results_dir + "/" + str(reaction_model) + "/potentialh.pvd")

    # evaluate and save to file L2-norm between solutions
    error = open(results_dir + "/" + str(reaction_model) + "/error_L2", 'w+')
    def evaluate_L2error(f1, f2):
        fi = interpolate(f1, Vh)
        m2, m3 = inner(fi - f2, fi - f2)*dx, inner(fi, fi)*dx
        error_rel	= abs(assemble(m2)/(assemble(m3)))
        error.write(str(t) + " " + str(error_rel) + "\n")
        return error_rel

    # function for apply external stimulus
    def apply_estimulus(b, estimulus, t, homogenized = 0):
        est_intensity, est_duration, delay, location = prms['phys'][estimulus]
        if homogenized == 0:
            if (t <= est_duration + dt + delay)*(t > delay):
               b = b + assemble(est_intensity*v*ds(location))
        else:
            if (t <= est_duration + dt + delay)*(t > delay):
                b = b + assemble(est_intensity*vh*dsh(location))
        return b

    # -----------------
    # --- TIME LOOP ---
    # -----------------
    t = dt; kkk = 4; start_time = time.time()
    print ' Solving Linear Systems: '
    while t <= T + dt:

        # update_gating_variables() [DEBIERA SER ASI]
        if reaction_model == 0:
            b, bh = 1/dt*M*phi0.vector(), 1/dt*Mh*phi0h.vector()

        if reaction_model == 1:            
            r.vector()[:] = (bb*phi0.vector() + r0.vector()/dt)/(1/dt +bb*d)
            rh.vector()[:] = (bb*phi0h.vector() + r0h.vector()/dt)/(1/dt +bb*d)
    
            r_file << r; rh_file << rh
            b  = 1/dt*M*phi0.vector()   + assemble(inner((c1*phi0*phi0   - c1*phi0*phi0*phi0    + c1*alpha*phi0*phi0   - c2*r  + phi0/dt), v)*dx)
            bh = 1/dt*Mh*phi0h.vector() + assemble(inner((c1*phi0h*phi0h - c1*phi0h*phi0h*phi0h + c1*alpha*phi0h*phi0h - c2*rh + phi0h/dt), vh)*dx)
            r0.assign(r); r0h.assign(rh)         

        b, bh = apply_estimulus(b, 'estimulo_1', t), apply_estimulus(bh, 'estimulo_1', t, 1)
        b, bh = apply_estimulus(b, 'estimulo_2', t), apply_estimulus(bh, 'estimulo_2', t, 1)

        # solve linear systems
        solver.solve(A,   phit.vector(),  b); solverh.solve(Ah, phith.vector(), bh)

        if kkk == 4:
            # calculate error between solutions
            error_rel = evaluate_L2error(phit, phith)
            # save solution to file
            phi_file << phi0; phih_file << phi0h; kkk = 0; 
        kkk = kkk + 1        

        # PRINT PROGRESS
        progress  = float(t)/float(T + dt)*100.0; time_elapsed = time.time() - start_time
        print '\r --> %.2f [ms]  |  %.0f %%  |  L2-error = %f  |  computing-time-elapsed = %.2f [seg] ' % (t, progress, error_rel, time_elapsed),
        sys.stdout.flush()

        # UPDATE POTENTIAL FIELD
        phi0.assign(phit)
        phi0h.assign(phith)
        t += dt
    error.close()
