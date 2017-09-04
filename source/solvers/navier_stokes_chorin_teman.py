# Begin demo

from dolfin import *

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
mesh = Mesh('../meshes/COARTED_60_h0.1.xml')

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

mu  = 0.035
dt  = 0.005
T   = 0.4
dim = 3
rho = 1

nu = mu/rho

# ******************************************
# Initialize sub-domain instances
inlet 		= Inlet()
outlet 		= Outlet()
noslip_bound    = No_slip()


# Mark boundaries of the domain
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
noslip_bound.mark(boundaries, 3)
inlet.mark(boundaries, 1)
outlet.mark(boundaries, 2)

# Define function spaces (P2-P1)
#V = VectorFunctionSpace(mesh, "Lagrange", 2)
V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Define boundary conditions
if dim == 3:
  #v_in    = Expression(("0.0", "-1*(x[0]*x[0] + x[2]*x[2] - 1)*sin(1/0.4*DOLFIN_PI*t)", "0.0"), t=0.0)
  noslip  = DirichletBC(V, (0, 0, 0), boundaries, 3)
  f       = Constant((0, 0, 0))
  # WOMERSLEY SOLUTIONFOR DC BC TODO: pasar todo esto a una funcion de python 
  LL 	= 2.5
  R 	= 2.0 # this is not the domain radius!
  omega  = (2*DOLFIN_PI/T)
  N_w 	= 1
  Pdrop   = -1877
  aa      = Pdrop/(rho*LL)
  sigma_e = mu*DOLFIN_PI*DOLFIN_PI/(rho*R*R)
  xmin = 1

  inflow    = "-30*(sin(DOLFIN_PI*((x[0]*x[0] + x[2]*x[2] - 1)/R*t)))"
  v_in = Expression(("0.0", inflow, "0.0"), t = 0, R = R)

 
else:
  #v_in    = Expression(("0.0", "1*(-x[0]*x[0] + 2*x[0])*sin(1/0.4*DOLFIN_PI*t)"), t=0.0)
  noslip  = DirichletBC(V, (0, 0), boundaries, 3)
  f 	  = Constant((0, 0))
  # WOMERSLEY SOLUTIONFOR DC BC TODO: pasar todo esto a una funcion de python 
  LL 	= 2.5
  R 	= 2.0 # this is not the domain radius!
  omega   = (2*DOLFIN_PI/T)
  N_w 	= 50
  Pdrop   = 1877.0
  aa      = Pdrop/(rho*LL)
  sigma_e = mu*DOLFIN_PI*DOLFIN_PI/(rho*R*R)

  inflow = ''
  for k in range(N_w):  
    kk = str(k)
    num     = "4*aa*((2*" + kk + " + 1)*(2*" + kk + " + 1)*sigma_e*sin(omega*t) - omega*cos(omega*t) + omega*exp(-1.0*((2*" + kk + "+1)*(2*" + kk + "+1))*sigma_e*t))"
    den     = "(DOLFIN_PI*(2*" + kk + "+1)*((2*" + kk + "+1)*(2*" + kk + "+1)*(2*" + kk + "+1)*(2*" + kk + "+1)*sigma_e*sigma_e +  omega*omega))"
    seno    = "(sin(DOLFIN_PI*(2*" + kk + " + 1)*(x[0])/R))"
    inflow = infl.ow + str('(') + num + str('/') + den + str(')')  + str('*') + seno

    if k != N_w-1: inflow = inflow + str('+')

inflow  = DirichletBC(V, v_in, boundaries, 1)
outflow = DirichletBC(Q, 0, boundaries, 2)
bcu = [noslip, inflow]
bcp = [outflow]

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

# Define coefficients
k = Constant(dt)

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
     nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# time dependent terms
#teman_1 = 0.5*(div(u0)*inner(u, v))*dx
#teman_2 = 0.5*(div(u1)*inner(u, v))*dx

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:

    # Update pressure boundary condition
    v_in.t = t

    # Compute tentative velocity step
    begin("Computing tentative velocity")
    #A1 = A1 + assemble(teman_1)
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "bicgstab", "default")
    end()

    # Pressure correction
    begin("Computing pressure correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    [bc.apply(p1.vector()) for bc in bcp]
    solve(A2, p1.vector(), b2, "bicgstab", prec)
    end()

    # Velocity correction
    begin("Computing velocity correction")
    #A3 = A3 + assemble(teman_2)
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "bicgstab", "default")
    end()
    
    ## Plot solution
    #plot(p1, title="Pressure", rescale=True)
    #plot(u1, title="Velocity", rescale=True)
    
    # Save to file
    ufile << u1
    pfile << p1

    # Move to next time step
    u0.assign(u1)
    t += dt
    print("t =", t)

# Hold plot
interactive()


