from dolfin import *
from functions.inout import readmesh

#h = 0.4
#mesh, subdomains, boundaries_MALO = readmesh('./meshes/COARTED_60_h' + str(h) + '.h5')
mesh = UnitSquareMesh(5,5)

P1 = VectorFunctionSpace(mesh, 'CG', 1)
P1_dim1  = FunctionSpace(mesh, 'CG', 1)
P2_dim1  = FunctionSpace(mesh, 'CG', 2)

ff = Function(P1_dim1)
alpha = Function(P2_dim1)

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

u_werp, v_werp = TestFunction(P1_dim1), TrialFunction(P1_dim1)
K_werp = assemble(inner(v_werp, u_werp)*dx)
diag_k = as_backend_type(K_werp).mat().getDiagonal().array
trace_k = diag_k.sum()
int_alpha = assemble(alpha*dx)

fine_mesh = refine(refine(refine(mesh)))
P1_fine = FunctionSpace(fine_mesh, "CG", 1)
alpha.set_allow_extrapolation(True)
alpha_fino = interpolate(alpha, P1_fine)

print "tr(M) = %g " % trace_k
print "int alpha dx = %g" % int_alpha

plot(alpha)
plot(alpha_fino)
interactive()