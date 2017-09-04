""" solver module """
from dolfin import *
import scipy.io as sio
import time
import numpy as np
import os.path
import sys


def compute(prms):
    ''' monodomain+minimal equation solver
        author: Felipe Galarce
        email: felipe.galarce.m@gmail.com
    '''

    from functions.inout import readmesh
    
    # Extract parameters
    results_dir = prms['io']['results']
    solve_EP, solve_HP = prms['io']['solve']

    dt 				= prms['num']['dt']
    sol_method, prec, solh_method, prech = prms['num']['krylov']
    deg				= prms['num']['deg']
    T 				= prms['num']['T']

    part 			= prms['phys']['part']
    sigma_1			= prms['phys']['sigma_1']
    est1_intensity, est1_delay, est1_duration = prms['phys']['est1']
    est2_intensity, est2_delay, est2_duration = prms['phys']['est2']
    phi_i, r_i, w_i, s_i 	= prms['phys']['initial_conditions']
    betha 			= prms['phys']['betha']; gamma = prms['phys']['gamma']
    amount_fibrosis 		= prms['phys']['amount_fibrosis']

    mesh_size			= prms['geo']['mesh_size']
    e1x, e1y 			= prms['geo']['lamination_direction_main']
    e2x, e2y 			= prms['geo']['lamination_direction_cross']   
    ffx, ffy 			= prms['geo']['main_fiber_direction']   
    ccx, ccy 			= prms['geo']['cross_fiber_direction']

    if prms['io']['xdmf'] == 1: # 
      sol = File(results_dir+"/sol.pvd")
      solh = File(results_dir+"/solh.pvd")
      error = open(results_dir + "/minimal/error_L2", 'w+')
      benchmark = open(results_dir + "/minimal/benchmark", 'w+')

    # Import minimal model parameters
    if 	 part == 'endo':
      mat = sio.loadmat('./input/minimal_parameters_endo.mat')
    elif part == 'epi':
      mat = sio.loadmat('./input/minimal_parameters_epi.mat')
    elif part == 'mid':
      mat = sio.loadmat('./input/minimal_parameters_mid.mat')
    else:      
      mat = sio.loadmat('./input/minimal_parameters_atrial.mat')      

    # TODO: traslate parameters to .yaml file
    p1_phi0 		= float(mat['p1_phi0'][0,0])
    p2_phiu 		= float(mat['p2_phiu'][0,0])
    p3_thetav		= float(mat['p3_thetav'][0,0])
    p4_thetaw		= float(mat['p4_thetaw'][0,0])
    p5_thetav_minus	= float(mat['p5_thetav_minus'][0,0])
    p6_theta_0		= float(mat['p6_theta_0'][0,0])
    p7_tauv1_minus	= float(mat['p7_tauv1_minus'][0,0])
    p8_tauv2_minus	= float(mat['p8_tauv2_minus'][0,0])
    p9_tauv_plus	= float(mat['p9_tauv_plus'][0,0])
    p10_tauw1_minus = float(mat['p10_tauw1_minus'][0,0])
    p11_tauw2_minus = float(mat['p11_tauw2_minus'][0,0])
    p12_kw_minus	= float(mat['p12_kw_minus'][0,0])
    p13_phiw_minus	= float(mat['p13_phiw_minus'][0,0])
    p14_tau_w_plus	= float(mat['p14_tau_w_plus'][0,0])
    p15_tau_fi		= float(mat['p15_tau_fi'][0,0])
    p16_tau_o1		= float(mat['p16_tau_o1'][0,0])
    p17_tau_o2		= float(mat['p17_tau_o2'][0,0])
    p18_tau_so1		= float(mat['p18_tau_so1'][0,0])
    p19_tau_so2		= float(mat['p19_tau_so2'][0,0])
    p20_k_so		= float(mat['p20_k_so'][0,0])
    p21_phi_so		= float(mat['p21_phi_so'][0,0])
    p22_tau_s1		= float(mat['p22_tau_s1'][0,0])
    p23_tau_s2		= float(mat['p23_tau_s2'][0,0])
    p24_ks		    = float(mat['p24_ks'][0,0])
    p25_phi_s		= float(mat['p25_phi_s'][0,0])
    p26_tau_si		= float(mat['p26_tau_si'][0,0])
    p27_tauw_inf	= float(mat['p27_tauw_inf'][0,0])
    p28_w_inf		= float(mat['p28_w_inf'][0,0])

    if prms['num']['dry'] == True:
        # perform DRY RUN (e.g. if FFC fails on cluster)
        T = -1
        prms['io']['vtk'] = 0
        print("PERFORMING DRY RUN")

    #TODO: sacar funciones a otro archivo
    def HS(x): # 	Heavy-Side function
      return (x > 0).astype(float)	

    class DiscontinuousTensor(Expression):# 	Class with methods to assign tensors to its respective materials
        def __init__(self, cell_function, tensors):
            self.cell_function = cell_function
            self.coeffs 	   = tensors
        def value_shape(self):
            return (2,2)
        def eval_cell(self, values, x, cell):
            subdomain_id = self.cell_function[cell.index]
            local_coeff  = self.coeffs[subdomain_id]
            local_coeff.eval_cell(values, x, cell)

    # Read mesh, boundaries and subdomains
#    if solve_EP:
#      mesh, subdomains, boundaries = readmesh('./meshes/fibrosis/' + mesh_size + '/' + amount_fibrosis + '/mesh.h5')
#    else:
#      mesh, subdomains, boundaries = readmesh('./meshes/fibrosis/dummy_mesh.h5')
#    if solve_HP:      
#      mesh_h, subdomains_h, boundaries_h = readmesh('./meshes/fibrosis/' + mesh_size + '/mesh_h.h5')
#    else:
#      mesh_h, subdomains_h, boundaries_h = readmesh('./meshes/fibrosis/dummy_mesh.h5')

    mesh, subdomains, boundaries       = readmesh('./meshes/fibro_3x3_1_10_40_50.h5')
    mesh_h, subdomains_h, boundaries_h = readmesh('./meshes/homogenized_3x3.h5')
    rank = mesh.mpi_comm().Get_rank()   

    if rank == 0:
      print mesh
      print mesh_h

    # caracterize fibrosis level
    if amount_fibrosis == 'high':
      theta_c = 0.4
      theta_f = 0.5
    elif amount_fibrosis == 'moderated':
      theta_c = 0.2
      theta_f = 0.5
    elif amount_fibrosis == 'healthy':
      theta_c = 0
      theta_f = 0
    elif amount_fibrosis == 'vertical_walls':
      theta_c = 1
      theta_f = 0.1

    # Define P1-Lagrange function spaces over meshes
    V    	= FunctionSpace(mesh,   "CG", prms['num']['deg'])
    Vh    	= FunctionSpace(mesh_h, "CG", prms['num']['deg'])						# function space for test and trial functions

    # ******************************************

    ff = np.matrix(str(ffx) + ';' + str(ffy)) 
    cc = np.matrix(str(ccx) + ';' + str(ccy))

    # TODO: some day re-do this section without use of numpy (replace all for ufl forms)
    # TODO: deduce lamination direction form fiber and cross fiber direction
    e1 = np.matrix(str(e1x) + ';' + str(e1y))
    e2 = np.matrix(str(e2x) + ';' + str(e2y))

    sigma_2, sigma_c = sigma_1/gamma, pow(10, -betha)*sigma_1 
    # healthy diffusion tensor
    Dh   = sigma_1*ff*ff.transpose() + sigma_2*cc*cc.transpose() 
    # collagen diffusion tensor
    Dcol = np.matrix(str(sigma_c) + ' 0; 0 ' + str(sigma_c))

    # Calculate homogenized tensor with rank 2 laminations
    corrector_numerador   = (theta_c*(1 - theta_c)*(Dcol - Dh)*e1)*((Dcol - Dh).transpose()*e1).transpose()
    corrector_denominador = (1 - theta_c)*(Dcol*e1).transpose()*e1  + theta_c*(Dh*e1).transpose()*e1;
    Deffp = theta_c*Dcol  + (1 - theta_c)*Dh  - corrector_numerador/corrector_denominador

    corrector_numerador = (theta_f*(1 - theta_f)*(Deffp - Dh)*e2)*((Deffp - Dh).transpose()*e2).transpose()
    corrector_denominador = (1 - theta_f)*(Deffp*e2).transpose()*e2 + theta_f*(Dh*e2).transpose()*e2
    Deff  = theta_f*Deffp + (1 - theta_f)*Dh - corrector_numerador/corrector_denominador

    # Save components of tensors in auxiliar variables in order to assign boundary conditions easyly
    Dh_11 = Dh[0,0]
    Dh_22 = Dh[1,1]
    Deff_11 = Deff[0,0]
    Deff_22 = Deff[1,1]
    
    # UFL version of the tensors
    Dh	= Constant(((sigma_1,      0),					
		    (0,      sigma_2))) 	
    Dcol = Constant(((sigma_c*1  , 0.0,),		
	             (0.0,    sigma_c*1)))

    Deff = Constant(((Deff[0,0], Deff[0,1]), 
		     (Deff[1,0], Deff[1,1])))
    
    # Assign tensors where they belongs    
    C = DiscontinuousTensor(subdomains, [Dh, Dcol])
 
    # ******************************************

    # fix number of cuadrature points
    q_degree = 3
    dx_ = dx(metadata={'quadrature_degree': q_degree})

    # Initial conditions
    phi1,  r1,  w1,  s1		= interpolate(Expression(str(phi_i)), V ), interpolate(Expression(str(r_i)), V ), interpolate(Expression(str(w_i)), V ), interpolate(Expression(str(s_i)), V) 
    phi1h, r1h, w1h, s1h	= interpolate(Expression(str(phi_i)), Vh), interpolate(Expression(str(r_i)), Vh), interpolate(Expression(str(w_i)), Vh), interpolate(Expression(str(s_i)), Vh)

    # ******************************************

    # Inward and outward currents factorization
    J, J_h	= Function(V), Function(Vh)

    # Continuous variational exact problem 
    u, v	= TrialFunction(V), TestFunction(V)
    a_K 	= inner(C * nabla_grad(u), nabla_grad(v)) * dx_ 
    a_M     	= u*v*dx_

    # Continuous Variational Homogenized Problem
    uh, vh  	= TrialFunction(Vh), TestFunction(Vh)
    a_Kh    	= inner(Deff * grad(uh), grad(vh)) * dx_ 
    a_Mh 	= uh*vh*dx_

    # Assemble stifness and mass matrix for discretized variational problem
    start = time.time()
    K   	= assemble(a_K)
    M   	= assemble(a_M)
    A  		= M  + dt*K
    end 	= time.time()
    time_assembly = end - start

    # Assemble stifness and mass matrix for discretized variational homogenized problem
    start 	= time.time()
    Kh   	= assemble(a_Kh)
    Mh   	= assemble(a_Mh)
    Ah  	= Mh  + dt*Kh
    end 	= time.time()
    time_assembly_h = end - start
    benchmark.write(str(time_assembly) + " " + str(time_assembly_h) + "\n")

    phi, phih, F_k, F_kh	= Function(V), Function(Vh), Function(V), Function(Vh)

    # ******************************************

    # Auxiliar vectors of the ufl-functions in order to get element-wise operations
    r1_arr,  w1_arr,  s1_arr	= r1.vector().array(), w1.vector().array(), s1.vector().array()
    r1h_arr, w1h_arr, s1h_arr	= r1h.vector().array(), w1h.vector().array(), s1h.vector().array()


    # Auxiliary functions to save and export re-scaled potential
    aux_out1, aux_out2		=	Function(V), Function(Vh)

    # ******************************************

    # External stimulus (the values of g have to be different to obtain the same stimula in both exact and homogenized tissue)
    est1_, est2_ = est1_intensity, est2_intensity
    est1h_, est2h_ = est1_intensity*(Deff_11/Dh_11), est2_intensity*(Deff_22/Dh_22)
    g1 		= Expression('est1_' , est1_=est1_)
    g2 		= Expression('est2_' , est2_=est2_)
    g1h 	= Expression('est1h_', est1h_=est1h_)
    g2h		= Expression('est2h_', est2h_=est2h_)
    ds   = Measure('ds', domain=mesh  , subdomain_data=boundaries  )
    ds_h = Measure('ds', domain=mesh_h, subdomain_data=boundaries_h)

    # ******************************************
    # Time - loop

    # Init Krylov solvers
#    solver, solverh = KrylovSolver(sol_method, prec), KrylovSolver(solh_method, prech)  
    # Reuse factorization
#    solver.parameters['preconditioner']['structure'] = 'same' 
#    solverh.parameters['preconditioner']['structure'] = 'same' 
    # Use previous time solution as initial guess
#    solver.parameters['nonzero_initial_guess'] = True
#    solverh.parameters['nonzero_initial_guess'] = True
    solver, solverh = PETScKrylovSolver("cg", "amg"), PETScKrylovSolver("cg", "amg")
    solver.set_reuse_preconditioner(True); solverh.set_reuse_preconditioner(True)
    solver.set_nonzero_guess(True);  solverh.set_nonzero_guess(True)    


    solver.parameters['report'] = False
    solverh.parameters['report'] = False
    #solver.parameters['monitor_convergence'] = True

    # the following prevent to evaluate conditional every time-step
    if prms['io']['xdmf'] == 1:
      counter	= 10 
      error_count = 100
    else:
      counter = 11    
      error_count = 101

    phi_i = Function(Vh)
    LI = LagrangeInterpolator() 

    t		= dt
    while t < T:
      if rank == 0:
	print "solving time step t = %g" % t 
      #***********************************
      # Solving EDO's

      # Update arrays
      phi1_arr	= phi1.vector().array()
      phi1h_arr	= phi1h.vector().array()
	
      eq1	= (1 - HS(phi1_arr - p5_thetav_minus)*p7_tauv1_minus + HS(phi1_arr - p5_thetav_minus)*p8_tauv2_minus);
      eq2	= p10_tauw1_minus + (p11_tauw2_minus - p10_tauw1_minus)*(1 + np.tanh(p12_kw_minus*(phi1_arr - p13_phiw_minus)))/2;
      eq3	= p18_tau_so1 + (p19_tau_so2 - p18_tau_so1)*(1 + np.tanh(p20_k_so*(phi1_arr - p21_phi_so)))/2;
      eq4	= (1 - HS(phi1_arr - p4_thetaw))*p22_tau_s1 + HS(phi1_arr - p4_thetaw)*p23_tau_s2;
      eq5	= (1 - HS(phi1_arr - p6_theta_0))*p16_tau_o1 + HS(phi1_arr - p6_theta_0)*p17_tau_o2;    
	
      r_inf	= (phi1_arr < p5_thetav_minus).astype(float);
      w_inf	= (1 - HS(phi1_arr - p6_theta_0))*(1 - phi1_arr/p27_tauw_inf) + HS(phi1_arr - p6_theta_0)*p28_w_inf;
	
      # Update gating variables
      r1_arr	= (r1_arr + (dt*r_inf*(1 - HS(phi1_arr - p3_thetav)))/eq1)/(1  +  (dt*(1 - HS(phi1_arr - p3_thetav)))/eq1 + (dt*HS(phi1_arr - p3_thetav))/p9_tauv_plus);
      w1_arr	= (w1_arr + (dt*w_inf*(1 - HS(phi1_arr - p4_thetaw)))/eq2)/(1  +  (dt*(1 - HS(phi1_arr - p4_thetaw)))/eq2 + (dt*HS(phi1_arr - p4_thetaw))/p14_tau_w_plus);
      s1_arr	= (s1_arr + (dt*(1 + np.tanh(p24_ks*(phi1_arr - p25_phi_s)))/2)/eq4) / (1 + dt/eq4);    

      # Current factorization
      Jfi_i	= -1/p15_tau_fi*r1_arr*HS(phi1_arr - p3_thetav) * (p2_phiu - phi1_arr + p3_thetav);
      Jfi_e	= -r1_arr*HS(phi1_arr - p3_thetav)*p3_thetav*p2_phiu/p15_tau_fi;
      Jso_i   	= (1 - HS(phi1_arr - p4_thetaw))/eq5;
      Jso_e	=  HS(phi1_arr - p4_thetaw)/eq3 - p1_phi0/eq5*(1 - HS(phi1_arr - p4_thetaw));
      Jsi_e	= -HS(phi1_arr - p4_thetaw)*w1_arr*s1_arr/p26_tau_si;
	
      J.vector()[:]= Jsi_e + Jso_e + Jfi_e + (Jfi_i + Jso_i)*phi1_arr
      
      # Solving EDO's (Homogenized Problem)
      eq1		= (1 - HS(phi1h_arr - p5_thetav_minus)*p7_tauv1_minus + HS(phi1h_arr - p5_thetav_minus)*p8_tauv2_minus);
      eq2		= p10_tauw1_minus + (p11_tauw2_minus - p10_tauw1_minus)*(1 + np.tanh(p12_kw_minus*(phi1h_arr - p13_phiw_minus)))/2;
      eq3		= p18_tau_so1 + (p19_tau_so2 - p18_tau_so1)*(1 + np.tanh(p20_k_so*(phi1h_arr - p21_phi_so)))/2;
      eq4		= (1 - HS(phi1h_arr - p4_thetaw))*p22_tau_s1 + HS(phi1h_arr - p4_thetaw)*p23_tau_s2;
      eq5		= (1 - HS(phi1h_arr - p6_theta_0))*p16_tau_o1 + HS(phi1h_arr - p6_theta_0)*p17_tau_o2;    
	
      r_inf		= (phi1h_arr < p5_thetav_minus).astype(float);
      w_inf		= (1 - HS(phi1h_arr - p6_theta_0))*(1 - phi1h_arr/p27_tauw_inf) + HS(phi1h_arr - p6_theta_0)*p28_w_inf;
	  
      # Update gating variables
      r1h_arr	= (r1h_arr + (dt*r_inf*(1 - HS(phi1h_arr - p3_thetav)))/eq1)/(1  +  (dt*(1 - HS(phi1h_arr - p3_thetav)))/eq1 + (dt*HS(phi1h_arr - p3_thetav))/p9_tauv_plus);
      w1h_arr	= (w1h_arr + (dt*w_inf*(1 - HS(phi1h_arr - p4_thetaw)))/eq2)/(1  +  (dt*(1 - HS(phi1h_arr - p4_thetaw)))/eq2 + (dt*HS(phi1h_arr - p4_thetaw))/p14_tau_w_plus);
      s1h_arr	= (s1h_arr + (dt*(1 + np.tanh(p24_ks*(phi1h_arr - p25_phi_s)))/2)/eq4) / (1 + dt/eq4);    

      # Current factorization
      Jfi_i		= -1/p15_tau_fi*r1h_arr*HS(phi1h_arr - p3_thetav) * (p2_phiu - phi1h_arr + p3_thetav);
      Jfi_e		= -r1h_arr*HS(phi1h_arr - p3_thetav)*p3_thetav*p2_phiu/p15_tau_fi;
      Jso_i   		= (1 - HS(phi1h_arr - p4_thetaw))/eq5;
      Jso_e		= HS(phi1h_arr - p4_thetaw)/eq3 - p1_phi0/eq5*(1 - HS(phi1h_arr - p4_thetaw));
      Jsi_e		= -HS(phi1h_arr - p4_thetaw)*w1h_arr*s1h_arr/p26_tau_si;
	
      J_h.vector()[:]= Jsi_e + Jso_e + Jfi_e + (Jfi_i + Jso_i)*phi1h_arr
      
      #**********************************  
      # Solving PDE
      
      if   (t >= est1_delay)*(t <= est1_delay + est1_duration):
        G		= assemble(g1*v *ds(1)  )
        Gh		= assemble(g1h*vh*ds_h(1))
      elif (t >= est2_delay)*(t <= est2_delay + est2_duration):
	    G		= assemble(g2*v *ds(4)  )
	    Gh		= assemble(g2h*vh*ds_h(4))
      else:
    	G, Gh = 0, 0;

      F_k.vector()[:] 	= J.vector()   
      b    		= M*phi1.vector()   - dt*M*F_k.vector()   + dt*G

      F_kh.vector()[:] 	= J_h.vector() 
      bh    		= Mh*phi1h.vector() - dt*Mh*F_kh.vector() + dt*Gh

      start 		= time.time()
      solver.solve(A , phi.vector(),  b)
      end 		= time.time(); time_solve   = end - start

      start 		= time.time()
      solverh.solve(Ah, phih.vector(), bh)  
      end 		= time.time(); time_solve_h = end - start

      phi1.assign(phi)
      phi1h.assign(phih)

      if error_count == 100:
	if solve_EP and solve_HP:
	  #**********************************
	  # Evaluate L2-norm.
	  LI.interpolate(phi_i, phi)
	  m2 		= inner(phi_i-phih, phi_i - phih)*dx
	  m3		= inner(phi_i, phi_i)*dx
	  l2_norm_dif 	= assemble(m2)
	  l2_norm	= assemble(m3)
	  error_rel	= abs(l2_norm_dif/(l2_norm + (t <= est1_delay)))
	  if rank == 0:
	    print "error = %g" % error_rel   
	  error.write(str(t) + " " + str(error_rel) + "\n")
	  error_count = 0
		  
      if counter == 10:
	#**********************************
	# Export solutions to VTK (re-scaled to physiologycal values)
	if part == 'atrial':
	  if solve_EP:
	    aux_out1.vector()[:] = 85.7*phi.vector()  - 80.9
	  if solve_HP:
	    aux_out2.vector()[:] = 85.7*phih.vector() - 80.9
	else:
	  if solve_EP:
	    aux_out1.vector()[:]	= 85.7*phi.vector()  - 84
	  if solve_HP:
	    aux_out2.vector()[:]	= 85.7*phih.vector() - 84
	sol   << aux_out1, t
	solh  << aux_out2, t 
    	counter = 0
      
      # PRINT PROGRESS
      progress  = float(t)/float(T + dt)*100.0
      print '\r --> %.2f [ms]  |  %.0f %%  |  L2-error = %f' % (t, progress, error_rel),
      sys.stdout.flush()
      
      benchmark.write(str(time_solve) + " " + str(time_solve_h) + "\n")  
      counter += 1
      error_count += 1      
      t = t + dt   
