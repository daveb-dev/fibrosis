from dolfin import *

def compute(h = 0.4, noise = 0, tau = 0.02, T = 0.4, realization = 0):
  import matplotlib.pyplot as plt
  from functions.inout import readmesh
  import numpy as np
  import csv
  import math

  # Class with subdomains to mark borders of the domain
  class Inlet(SubDomain):
      def inside(self, x, on_boundary):
	return x[1] < DOLFIN_EPS
  class Outlet(SubDomain):
      def inside(self, x, on_boundary):
	  return x[1] > 5 - 5*DOLFIN_EPS
  class No_slip(SubDomain):
      def inside(self, x, on_boundary):
	  return on_boundary and x[1] <= 5 - DOLFIN_EPS and x[1] >= 0.0

  # ===========================
  # Input parameters
  mu    = 0.035
  rho   = 1
  dt    = 0.002	

  # set-up methods to be calculated
  DAE		= 1
  DAEi		= 0
  PIMRP		= 1
  BIMRP		= 0
  PPE 		= 1
  WERP		= 1
  STE 		= 1
  STEi		= 1
  test_brinkman_viscocity = 0
  test_viscocity_contribution = 0

  if test_viscocity_contribution:
    test_viscocity_contribution_array = []
    test_viscocity_contribution_file = open('./results/test_viscocity_contribution.csv', 'w+')

  # set viscocity for brinkman test functions
  optimal_mu = 0.35

  results           = './results/pressure_drop'
  velocity_measures = './results/velocity_measures/h' + str(h)
  test_functions    = './results/test_functions'
  p_drop_file       = './results/pressure_drop/all/p_drop_file_h' + str(h) +'_tau' + str(tau) +'_noise' + str(noise) + '_realization' + str(realization) + '.csv'
  sigma	    	    = 0.1*280*noise	# variance of the additive noise in the sample

  #============================
  # Load Mesh
  mesh, subdomains, boundaries_MALO = readmesh('./meshes/COARTED_60_h' + str(h) + '.h5')

  # ******************************************
  # Initialize sub-domain instances
  inlet 	= Inlet()
  outlet 	= Outlet()
  noslip_bound  = No_slip()

  # Mark boundaries of the domain
  boundaries = FacetFunction("size_t", mesh)
  boundaries.set_all(3)
  noslip_bound.mark(boundaries, 2)
  inlet.mark(boundaries, 1)
  outlet.mark(boundaries, 0)

  ds, n = Measure('ds', domain=mesh  , subdomain_data=boundaries), FacetNormal(mesh)

  #============================
  # Load measures P1-data  

  P1	= VectorFunctionSpace(mesh, "CG", 1)
  if noise == 1:
    u_measures_file = HDF5File(mesh.mpi_comm(), velocity_measures + '/velocity_noise_realization' + str(realization) + '.h5', "r")
  else:
    u_measures_file = HDF5File(mesh.mpi_comm(), velocity_measures + "/velocity.h5", "r")

  ut, u0, u_, dudt = Function(P1), Function(P1), Function(P1), Function(P1)

  #============================
  # Load Stokes test functions data
  B          = VectorFunctionSpace(mesh, 'Bubble', 4)
  V1         = P1 + B

  TF_stokes  = Function(V1)
  u_st_file1 = HDF5File(mesh.mpi_comm(), test_functions + "/stokes_h" + str(h) + "_P1bP1/velocity.h5", "r")
  u_st_file1.read(TF_stokes, "/velocity")

  #============================
  # Load Brinkman test functions data
  if test_brinkman_viscocity:
    mumu = []
    for i in range(5,0,-1):
      mumu.append(0.035*pow(10, -i))
    for i in range(5):
      mumu.append(0.035*pow(10, i))
    TF_brinkman  = Function(V1)
    brinkman_file = open('./results/pressure_drop/all/brinkman/brinkman_viscocity_test.csv', 'w+')
    brinkman_file.write('999, ')
    for i in range(len(mumu)):
      brinkman_file.write(str(mumu[i]) + ',')
    brinkman_file.write('\n')
  else:
    TF_brinkman  = Function(V1)
    u_brinkman_file = HDF5File(mesh.mpi_comm(), test_functions + "/brinkman_h" + str(h) + "_mu" + str(optimal_mu) + "_P1bP1/velocity.h5", "r")
    u_brinkman_file.read(TF_brinkman, "/velocity")

  # ===========================
  # OUTPUT FILE with pressure drop estimators data
  press_drop_file = open(p_drop_file, 'w+')  
  time_array, p_drop_array = [], []
  t, k = tau, 1

  # ===========================
  # Configure Solvers
  # PPE
  Q1 = FunctionSpace(mesh, 'CG', 1)
  q_ppe1, p_ppe1 = TestFunction(Q1), TrialFunction(Q1)
  bc_ppe1	= DirichletBC(Q1, Constant(0.0), boundaries, 1)
  p_ppe1_sol = Function(Q1)
  a_ppe1 = inner(grad(p_ppe1), grad(q_ppe1))*dx
  L_ppe1 =  - rho/tau*inner(ut - u0, grad(q_ppe1))*dx \
	    - rho*inner(grad(u_)*u_, grad(q_ppe1))*dx \
	    + mu*inner(div(grad(u_)), grad(q_ppe1))*dx

  solver1 = KrylovSolver('cg', 'amg')
  solver1.parameters['report'] = False
  solver1.parameters['preconditioner']['structure'] = 'same'
  solver1.parameters['nonzero_initial_guess'] = True

  A_ppe1, b_ppe1 = PETScMatrix(), PETScVector()

  # WERP
  u_werp, v_werp = TestFunction(P1), TrialFunction(P1)
  K_werp = assemble(mu*inner(grad(v_werp), grad(u_werp))*dx)
  diag_k = as_backend_type(K_werp).mat().getDiagonal().array
  trace_k = diag_k.sum()

  def Ekin(u):
    return rho/2.0/tau*inner(u, u)*dx
  def Econv(u):
    return rho/2.0*inner(u, n)*inner(u, u)*ds(0) + rho/2.0*inner(u, n)*inner(u, u)*ds(1)
  def Evisc(u):
    return mu*inner(grad(u), grad(u))*dx \
	    - mu*inner(grad(u)*n, u)*ds(0) \
	    - mu*inner(grad(u)*n, u)*ds(1) \
	    - mu*inner(grad(u)*n, u)*ds(2)

  # STE and STEi
  W_ste1	= V1*Q1

  w_ste1, p_ste1_sol, aux_ste1 = Function(W_ste1), Function(Q1), Function(V1)
  w_stei1, p_stei1_sol, aux_stei1 = Function(W_ste1), Function(Q1), Function(V1)

  (u_ste1, p_ste1) = TrialFunctions(W_ste1)
  (v_ste1, q_ste1) = TestFunctions(W_ste1)

  solver_ste1  = LinearSolver('mumps'); solver_ste1.parameters['report'] = False;
  solver_stei1 = LinearSolver('mumps'); solver_stei1.parameters['report'] = False

  # Set PETSc MUMPS paramter (this is required to prevent a memory error
  # in some cases when using MUMPS LU solver).
  if has_petsc():
    PETScOptions.set("mat_mumps_icntl_14", 40.0)

  bc_ste1 = DirichletBC(W_ste1.sub(0), project(Expression(("0.0","0.0", "0.0"), degree=3), V1), 'on_boundary')

  A_ste1, b_ste1 = PETScMatrix(), PETScVector()
  A_stei1, b_stei1 = PETScMatrix(), PETScVector()

  # DAE and DAEi
  V_RT1 = FunctionSpace(mesh, "RT" , 1)

  Mixed_RT1 = V_RT1*Q1

  (u_dae1, p_dae1) = TrialFunctions(Mixed_RT1)
  (v_dae1, q_dae1) = TestFunctions(Mixed_RT1)

  solver_dae1 = LinearSolver('mumps')
  solver_dae1.parameters['report'] = False

  solver_daei1 = LinearSolver('mumps')
  solver_daei1.parameters['report'] = False

  A_dae1,  b_dae1 = PETScMatrix(), PETScVector()
  A_daei1, b_daei1 = PETScMatrix(), PETScVector()

  a_dae1 	=   inner(u_dae1, v_dae1)*dx \
		- p_dae1*div(v_dae1)*dx \
		+ div(u_dae1)*q_dae1*dx

  L_dae1	= - rho/tau*inner(ut - u0, v_dae1)*dx \
		- rho*inner(grad(u_)*u_, v_dae1)*dx \
		+ mu*inner(div(grad(u_)), v_dae1)*dx

  L_daei1	=   - rho/tau*inner(ut - u0, v_dae1)*dx \
		+ rho*inner(grad(v_dae1)*u_, u_)*dx \
		- rho*inner(u_,n)*inner(u_, v_dae1)*ds(0) - rho*inner(u_,n)*inner(u_, v_dae1)*ds(1) - rho*inner(u_,n)*inner(u_, v_dae1)*ds(2) \
		- mu*inner(grad(u_), grad(v_dae1))*dx \
		+ mu*inner(grad(u_)*n, v_dae1)*ds(0) + mu*inner(grad(u_)*n, v_dae1)*ds(1) + mu*inner(grad(u_)*n, v_dae1)*ds(2)

  w_dae1, p_dae_sol1, aux_dae1 = Function(Mixed_RT1), Function(Q1), Function(V_RT1)
  w_daei1, p_daei_sol1, aux_daei1 = Function(Mixed_RT1), Function(Q1), Function(V_RT1)

  class BoundarySource(Expression):
	def __init__(self, mesh):
	    self.mesh = mesh
	def eval_cell(self, values, x, ufc_cell):
	    cell = Cell(self.mesh, ufc_cell.index)
	    n = cell.normal(ufc_cell.local_facet)
	    g = 0
	    values[0] = g*n[0]
	    values[1] = g*n[1]
	    values[2] = g*n[2]
	def value_shape(self):
	    return (3,)
  G = BoundarySource(mesh)

  bc_dae1 = DirichletBC(Mixed_RT1.sub(0), G, 'on_boundary')

  p_drop_pimrp_P1bP1 = 0
  p_drop_bimrp_P1bP1 = 0
  p_drop_ppe_P1 = 0
  p_drop_ste_P1bP1 = 0
  p_drop_stei_P1bP1 = 0
  p_drop_werp = 0
  p_drop_cwerp = 0
  p_drop_dae_RT1P1 = 0
  p_drop_daei_RT1P1 = 0

  def calculate_p_drop(p):
    return -1/DOLFIN_PI*(assemble(p*ds(1)) - assemble(p*ds(0)))

  # Calculate IMRP pressure drop estimation for any test function TF
  def IMRP(TF):
    lambda_ = assemble(inner(TF,  n)*ds(1))
    PDROP   = assemble(rho*inner(dudt, TF)*dx - rho*inner(grad(TF)*u_, u_)*dx + mu*inner(grad(u_), grad(TF))*dx   \
		      + rho*inner(u_ ,n)*inner(u_, TF)*ds(0) - mu*inner(grad(u_)*n, TF)*ds(0) \
		      + rho*inner(u_ ,n)*inner(u_, TF)*ds(1) - mu*inner(grad(u_)*n, TF)*ds(1))/lambda_
    return PDROP

  A_ste = assemble(inner(grad(u_ste1), grad(v_ste1))*dx - p_ste1*div(v_ste1)*dx + div(u_ste1)*q_ste1*dx)
  L_ste1= - rho/tau*inner(ut - u0, v_ste1)*dx \
	    - rho*inner(grad(u_)*u_, v_ste1)*dx \
	    + mu*inner(div(grad(u_)), v_ste1)*dx

  L_stei1	= - rho/tau*inner(ut - u0, v_ste1)*dx \
		  + rho*inner(grad(v_ste1)*u_, u_)*dx \
		  - mu*inner(grad(u_), grad(v_ste1))*dx

  if test_viscocity_contribution:
    L_stei1	= mu*inner(grad(u_), grad(v_ste1))*dx
    L_ppe1 	= mu*inner(div(grad(u_)), grad(q_ppe1))*dx

  import time
  while t <= T:
    print "Time: %g [seg]" % t
    u_measures_file.read(u0, "velocity_" + str(t - tau))
    u_measures_file.read(ut, "velocity_" + str(t))

    ut_vec, u0_vec, dudt_vec	= ut.vector(), u0.vector(), dudt.vector()
    u_vec   		= (ut_vec + u0_vec)/2
    dudt_vec		= 2*(u_vec - u0_vec)/tau
    u_.vector()[:], dudt.vector()[:] = u_vec, dudt_vec

    # =====================================
    # Method #1: P-IMRP 
    if PIMRP:
      p_drop_pimrp_P1bP1 = IMRP(TF_stokes)

    # =====================================
    # Method #2: B-IMRP
    if BIMRP:
      if test_brinkman_viscocity:
	brinkman_file.write(str((t - tau/2)) + ', ')
	for i in range(len(mumu)):
	  print "calculating for mu = %g \t poise" % mumu[i]
	  u_brinkman_file = HDF5File(mesh.mpi_comm(), test_functions + "/brinkman_h" + str(h) + "_mu" + str(mumu[i]) + "_P1bP1/velocity.h5", "r")
	  u_brinkman_file.read(TF_brinkman, "/velocity")	
	  p_drop_bimrp_P1bP1 = IMRP(TF_brinkman)
	  brinkman_file.write(str(p_drop_bimrp_P1bP1*0.000750061505) + ", ")
	brinkman_file.write('\n')
      else:
	p_drop_bimrp_P1bP1 = IMRP(TF_brinkman)

    # =====================================
    # Method #3: PPE
    if PPE:
      assemble_system(a_ppe1, L_ppe1, bc_ppe1, A_tensor=A_ppe1, b_tensor=b_ppe1)
      solver1.solve(A_ppe1, p_ppe1_sol.vector(), b_ppe1)
      p_drop_ppe_P1 = calculate_p_drop(p_ppe1_sol)

    # =====================================
    # Method #4: WERP
    if WERP:
      lambda_werp = assemble(inner(u_, n)*ds(1))
      p_drop_werp = -1/lambda_werp*assemble(Ekin(ut) - Ekin(u0) + Econv(u_) + Evisc(u_))
      if noise:
	# =====================================
	# Method #5: cWERP
	p_drop_cwerp = p_drop_werp + 1.0/lambda_werp*sigma*sigma/2.0*trace_k

    # =====================================
    #  Method #6: STE
    if STE:
      b_ste = assemble(L_ste1)
      A_ste_2 = A_ste
      bc_ste1.apply(A_ste_2, b_ste)
      solver_ste1.solve(A_ste_2, w_ste1.vector(), b_ste)
      aux_ste1, p_ste1_sol = w_ste1.split(deepcopy = True) 
      p_drop_ste_P1bP1 = calculate_p_drop(p_ste1_sol)

    # =====================================
    # Method #7: STE-int
    if STEi:
      b_stei = assemble(L_stei1)
      bc_ste1.apply(A_ste, b_stei)
      solver_ste1.solve(A_ste, w_stei1.vector(), b_stei)
      aux_stei1, p_stei1_sol = w_stei1.split(deepcopy = True)
      p_drop_stei_P1bP1 = calculate_p_drop(p_stei1_sol)

    # =====================================
    # Method #8: DAE
    if DAE:
      assemble_system(a_dae1, L_dae1, bc_dae1, A_tensor=A_dae1, b_tensor=b_dae1)
      solver_dae1.solve(A_dae1, w_dae1.vector(), b_dae1)
      aux_dae1, p_dae_sol1 = w_dae1.split(deepcopy = True)
      p_drop_dae_RT1P1 = calculate_p_drop(p_dae_sol1)

    # =====================================
    # Method #9: DAEi
    if DAEi:
      assemble_system(a_dae1, L_daei1, bc_dae1, A_tensor=A_daei1, b_tensor=b_daei1)
      solver_daei1.solve(A_daei1, w_daei1.vector(), b_daei1)  
      aux_daei1, p_daei_sol1 = w_daei1.split(deepcopy = True)  
      p_drop_daei_RT1P1 = calculate_p_drop(p_daei_sol1)

    print " -- > relative pressure gradient" 
    print "\tPPE:"
    print "\t\tP1\t= %g \t" % p_drop_ppe_P1
    print "\tPIMRP:"  
    print "\t\tP1bP1\t= %g \t" % p_drop_pimrp_P1bP1 
    print "\tBIMRP:"
    print "\t\t P1bP1\t= %g \t" % p_drop_bimrp_P1bP1 
    print "\tSTE:"
    print "\t\tP1bP1\t= %g \t" % p_drop_ste_P1bP1
    print "\tSTEi:"  
    print "\t\tP1bP1\t= %g \t" % p_drop_stei_P1bP1
    print "\tWERP \t\t= %g \t" % p_drop_werp
    if noise == 1:
      print "\tcWERP \t\t= %g \t" % p_drop_cwerp      
    print "\tDAE:"  
    print "\t\tRT1P1\t= %g \t" % p_drop_dae_RT1P1
    print "\tDAEi:"  
    print "\t\tRT1P1\t= %g \t" % p_drop_daei_RT1P1

    # pushback pressure drop results to array
    p_drop_array.append([-p_drop_pimrp_P1bP1, \
			    -p_drop_bimrp_P1bP1,
			    -p_drop_ppe_P1, \
			    p_drop_werp, \
			    p_drop_cwerp, \
			    -p_drop_ste_P1bP1, \
			    -p_drop_stei_P1bP1, \
			    -p_drop_dae_RT1P1, \
			    -p_drop_daei_RT1P1])
    time_array.append((t*1000 - tau*1000/2))

    # save pressure drop values to file
    press_drop_file.write(str(t*1000 - tau*1000/2) + ",")
    for i in range(len(p_drop_array[0])):
      press_drop_file.write(str(np.array(p_drop_array)[np.array(p_drop_array).shape[0] - 1 , i]*0.000750061505) + ",")
    press_drop_file.write("\n")

    # update velocity measures
    t, k = t + tau, k + 1
  if test_brinkman_viscocity:
    brinkman_file.close()
  u_measures_file.close(); 
  return np.array(time_array), np.array(p_drop_array)*0.000750061505 # change units to mmHg (more suitable)