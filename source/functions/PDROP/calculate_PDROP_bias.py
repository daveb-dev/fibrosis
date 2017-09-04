from functions.inout import readmesh
from dolfin import *
import os.path
import sys
sys.setrecursionlimit(9999999)

# Set PETSc MUMPS paramter (this is required to prevent a memory error in some cases when using MUMPS LU solver).
if has_petsc():
  PETScOptions.set("mat_mumps_icntl_14", 40.0)

def calculate_p_drop(p):
  return 1/DOLFIN_PI*(assemble(p*ds(1)) - assemble(p*ds(0)))

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

h		= 0.1
rho		= 1
sigma		= 28
mu		= 0.035
test_functions  = './results/test_functions'
alpha_file = './input/alpha_h' + str(h) + '.h5'

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

P1 = VectorFunctionSpace(mesh, 'Lagrange',     1)
P2 = VectorFunctionSpace(mesh, 'Lagrange', 2)
B  = VectorFunctionSpace(mesh, 'Bubble', 4)
Q  = FunctionSpace(mesh, 'Lagrange', 1)
V  = P1 + B


P1_dim1 = FunctionSpace(mesh, 'Lagrange', 1)
P2_dim1 = FunctionSpace(mesh, 'Lagrange', 2)
B_dim1  = FunctionSpace(mesh, 'Bubble', 4)
#V_dim1  = P1_dim1 + B_dim1

alpha = Function(P2_dim1)

if os.path.isfile(alpha_file):
  alpha_hdf5 = HDF5File(mesh.mpi_comm(), alpha_file, 'r')
  alpha_hdf5.read(alpha, "alpha")
else:
# Generate alpha function if is not already created
  ff    = Function(P1_dim1)
  ndofs = ff.vector().array().size
  for i in range(ndofs):
    print 'looping over dof # %g of %g' % (i, ndofs)

    ff_vec = ff.vector().array()
    ff_vec[i] = 1
    ff.vector()[:] = ff_vec


    tmp = interpolate(ff, P2_dim1)
    tmp_arr = tmp.vector().array()

    alpha_vec = alpha.vector().array()
    tmp_arr *= tmp_arr
    alpha_vec += tmp_arr
    alpha.vector()[:] = alpha_vec

    ff_vec         = ff.vector()
    ff_vec_arr     = ff_vec.array()
    ff_vec_arr[i]  = 0
    ff.vector()[:] = ff_vec_arr

  alpha_hdf5 = HDF5File(mesh.mpi_comm(), alpha_file, 'w')
  alpha_hdf5.write(alpha, "alpha")
  alpha_hdf5.close()

# Load stokes functions
v  = Function(V)
u_st_file = HDF5File(mesh.mpi_comm(), test_functions + "/stokes_h" + str(h) + "_P1bP1/velocity.h5", "r")
u_st_file.read(v, "/velocity")

# ============================
#	IMRP
lambda_v = assemble(inner(v,  n)*ds(1))
n    = FacetNormal(mesh)
bias_imrp = rho*sigma*sigma/(2*lambda_v)*(-assemble(alpha*div(v)*dx) + assemble(alpha*inner(v, n)*ds(0) + alpha*inner(v, n)*ds(1) + alpha*inner(v, n)*ds(2)))*0.000750061505

# ============================
#	PPE
q_ppe1, p_ppe1 = TestFunction(Q), TrialFunction(Q)
bc_ppe1	= DirichletBC(Q, Constant(0.0), boundaries, 1)
p_ppe1_sol = Function(Q)
a_ppe1 = inner(grad(p_ppe1), grad(q_ppe1))*dx
L_ppe1 = -rho*sigma*sigma/4.0*inner(grad(alpha), grad(q_ppe1))*dx
A_ppe1, b_ppe1 = PETScMatrix(), PETScVector()
assemble_system(a_ppe1, L_ppe1, bc_ppe1, A_tensor=A_ppe1, b_tensor=b_ppe1)
solver1 = KrylovSolver('cg', 'amg')
solver1.parameters['report'] = False
solver1.parameters['preconditioner']['structure'] = 'same'
solver1.parameters['nonzero_initial_guess'] = True
solver1.solve(A_ppe1, p_ppe1_sol.vector(), b_ppe1)
bias_ppe = calculate_p_drop(p_ppe1_sol)*0.000750061505

# ============================
#	STE
W_ste1	= V*Q
(u_ste1, bias_ste) = TrialFunctions(W_ste1)
(v_ste1, q_ste1) = TestFunctions(W_ste1)
solver_ste1  = LinearSolver('mumps'); solver_ste1.parameters['report'] = False;
bc_ste1 = DirichletBC(W_ste1.sub(0), project(Expression(("0.0","0.0", "0.0"), degree=3), V), 'on_boundary')
A_ste1, b_ste = PETScMatrix(), PETScVector()
w_ste1, bias_ste_sol, aux_ste1 = Function(W_ste1), Function(Q), Function(V)
A_ste = assemble(inner(grad(u_ste1), grad(v_ste1))*dx - bias_ste*div(v_ste1)*dx + div(u_ste1)*q_ste1*dx)
b_ste = assemble(-rho*sigma*sigma/4.0*inner(grad(alpha), v_ste1)*dx)
A_ste2 = A_ste
bc_ste1.apply(A_ste2, b_ste)
solver_ste1.solve(A_ste2, w_ste1.vector(), b_ste)
aux_ste1, bias_ste_sol = w_ste1.split(deepcopy = True) 
bias_ste = calculate_p_drop(bias_ste_sol)*0.000750061505

# ============================
#	STEi
w_stei1, p_stei1_sol, aux_stei1 = Function(W_ste1), Function(Q), Function(V)
solver_stei1 = LinearSolver('mumps'); solver_stei1.parameters['report'] = False
bc_ste1 = DirichletBC(W_ste1.sub(0), project(Expression(("0.0","0.0", "0.0"), degree=3), V), 'on_boundary')
A_ste1, b_ste1 = PETScMatrix(), PETScVector()
A_stei1, b_stei1 = PETScMatrix(), PETScVector()
L_stei1 = -rho*sigma*sigma/2*alpha*div(v_ste1)*dx
b_stei = assemble(L_stei1)
bc_ste1.apply(A_ste, b_stei)
solver_ste1.solve(A_ste, w_stei1.vector(), b_stei)
aux_stei1, p_stei1_sol = w_stei1.split(deepcopy = True)
bias_stei = calculate_p_drop(p_stei1_sol)*0.000750061505

# ============================
#	DAE
V_RT1 = FunctionSpace(mesh, "RT" , 1)
Mixed_RT1 = V_RT1*Q
(u_dae1, p_dae1) = TrialFunctions(Mixed_RT1)
(v_dae1, q_dae1) = TestFunctions(Mixed_RT1)
w_dae1, p_dae_sol1, aux_dae1 = Function(Mixed_RT1), Function(Q), Function(V_RT1)
solver_dae1 = LinearSolver('mumps')
solver_dae1.parameters['report'] = False
A_dae1,  b_dae1 = PETScMatrix(), PETScVector()
A_dae1 = assemble(inner(u_dae1, v_dae1)*dx \
	      - p_dae1*div(v_dae1)*dx \
	      + div(u_dae1)*q_dae1*dx)
b_dae1 = assemble(-rho*sigma*sigma/4.0*inner(grad(alpha), v_dae1)*dx)
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
bc_dae1.apply(A_dae1, b_dae1)
solver_dae1.solve(A_dae1, w_dae1.vector(), b_dae1)
aux_dae1, p_dae_sol1 = w_dae1.split(deepcopy = True)
bias_dae = calculate_p_drop(p_dae_sol1)*0.000750061505

# ============================
#	WERP
u0, ut, u_ = Function(P1), Function(P1), Function(P1)
u_measures_file = HDF5File(mesh.mpi_comm(), './results/velocity_measures/h' + str(h) + '/velocity_noise_realization50.h5', "r")
u_measures_file.read(u0, "velocity_" + str(0.26))
u_measures_file.read(ut, "velocity_" + str(0.24))
ut_vec, u0_vec = ut.vector(), u0.vector()
u_vec   = (ut_vec + u0_vec)/2
u_.vector()[:]= u_vec
u_werp, v_werp = TestFunction(P1), TrialFunction(P1)
K_werp = assemble(mu*inner(grad(v_werp), grad(u_werp))*dx)
diag_k = as_backend_type(K_werp).mat().getDiagonal().array
trace_k = diag_k.sum()
bias_werp = -1/assemble(inner(u_, n)*ds(1))*(sigma*sigma/2*trace_k)*0.000750061505


# ============================
#	PRINT RESULTS
print "\n\n\t=== BIAS for h = % g [cm] ===\n" % h
print "\tIMRP \t= %g \t[mmHg]" % bias_imrp
print "\tPPE \t= %g \t[mmHg]" % bias_ppe
print "\tSTE \t= %g \t[mmHg]" % bias_ste
print "\tSTEi\t= %g \t[mmHg]" % bias_stei
print "\tDAE \t= %g \t[mmHg]" % bias_dae
print "\tWERP \t= %g \t[mmHg]" % bias_werp
print "\n"

u_werp, v_werp = TestFunction(P1_dim1), TrialFunction(P1_dim1)
K_werp = assemble(inner(v_werp, u_werp)*dx)
diag_k = as_backend_type(K_werp).mat().getDiagonal().array
trace_k = diag_k.sum()
int_alpha = assemble(alpha*dx)
print "\n\nNOTE: Check is alpha function is acurrately enough using the identity: tr(M) = int alpha dx"
print "int alpha dx =  %g" % int_alpha
print "tr(M) = %g" % trace_k