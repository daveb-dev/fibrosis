# compare Subdomains vs Expression
# http://fenicsproject.org/documentation/tutorial/materials.html

from dolfin import *
import mshr
import scipy.linalg as la
import numpy as np

set_log_level(1)


# Define Dirichlet boundary (x = 0 or x = 1)
class BoundaryLeft(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < DOLFIN_EPS and on_boundary


class BoundaryRight(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 1 - DOLFIN_EPS and on_boundary


class Omega0(SubDomain):
    def inside(self, x, on_boundary):
        return True if x[1] <= 0.5 else False


class Omega1(SubDomain):
    def inside(self, x, on_boundary):
        return True if x[1] >= 0.5 else False

N = 5
e1, e2, e3, e4 = [], [], [], []
E1, E2, E3, E4 = [], [], [], []
for N in range(3, 8):
    # mesh = UnitSquareMesh(2**N, 2**N)
    domain = mshr.Rectangle(Point(0., 0.), Point(1., 1.))
    mesh = mshr.generate_mesh(domain, 2**N)

    V = FunctionSpace(mesh, "CG", 1)
    V0 = FunctionSpace(mesh, "DG", 0)

    left = BoundaryLeft()
    right = BoundaryRight()
    bcs = [DirichletBC(V, 0., left),
           DirichletBC(V, 1., right)]

    subdomains = CellFunction("size_t", mesh)
    subdomains.set_all(0)
# Mark subdomains with numbers 0 and 1
    subdomain0 = Omega0()
    subdomain0.mark(subdomains, 0)
    subdomain1 = Omega1()
    subdomain1.mark(subdomains, 1)
    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)

    k = Function(V0)
    k0 = 1.5
    k1 = 50
    k_values = [k0, k1]  # values of k in the two subdomains
    tmp = np.asarray(subdomains.array(), dtype=np.int32)
    k.vector()[:] = np.choose(tmp, k_values)

# Exact solution
    u_ex = Expression("x[1] <= 0.5 ? 2*x[1]*k1/(k0+k1) : \
            ((2*x[1]-1)*k0+k1)/(k0+k1)", k0=k0, k1=k1)


# Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(0)
    L = f*v*dx

# Var 1: Subdomains "old"  (tutorial/materials)
# NOTE: 1 and 2 seem always to give the same results
    a = inner(k*grad(u), grad(v))*dx

    u1 = Function(V)
    solve(a == L, u1, bcs)

# Var 2: Subdomains "new" (demo 20)
    a0 = Constant(k0)
    a1 = Constant(k1)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(a0*grad(u), grad(v))*dx(0) + inner(a1*grad(u), grad(v))*dx(1)
    u2 = Function(V)
    solve(a == L, u2, bcs)

# Var 3: evaluate expression
# NOTE: Behaves as if "kf" was interpolated onto CG1 elements, i.e. linear
# interpolation of the discontinuity
    kf = Expression("x[1]<=0.5 ? k0 : k1", k0=k0, k1=k1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(kf*grad(u), grad(v))*dx
    u3 = Function(V)
    solve(a == L, u3, bcs)

# Var 4: interpolate expression
# NOTE: on structured "UnitSquareMesh" same results as 1, 2;
# on unstructured mesh smaller error norm for fine grids
# with V2 elements significantly more accurate
    ki = interpolate(kf, V0)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(ki*grad(u), grad(v))*dx
    u4 = Function(V)
    solve(a == L, u4, bcs)

# plot solutions
    # plot(u_ex, mesh, title="u exact")
    # plot(u1, title="u1, interp. R")
    # plot(u2, title="u2, expression")
    # plot(u3, title="u3, subdomain")

    u0 = interpolate(u_ex, V)
    v0 = u0.vector()
    v1 = u1.vector()
    v2 = u2.vector()
    v3 = u3.vector()
    v4 = u4.vector()

    # e1.append(norm(v1-v0)/norm(v0))
    # e2.append(norm(v2-v0)/norm(v0))
    # e3.append(norm(v3-v0)/norm(v0))
    # e4.append(norm(v4-v0)/norm(v0))

    e1.append(errornorm(u1, u0)/norm(u0))
    e2.append(errornorm(u2, u0)/norm(u0))
    e3.append(errornorm(u3, u0)/norm(u0))
    e4.append(errornorm(u4, u0)/norm(u0))

    a0 = v0.array()
    a1 = v1.array()
    a2 = v2.array()
    a3 = v3.array()
    a4 = v4.array()

    E1.append(la.norm(a1-a0)/la.norm(a0))
    E2.append(la.norm(a2-a0)/la.norm(a0))
    E3.append(la.norm(a3-a0)/la.norm(a0))
    E4.append(la.norm(a4-a0)/la.norm(a0))

print("DOLFIN errornorm()")
for i in range(len(e1)):
    print("%i^2:\t %.3e,\t %.3e,\t %.3e,\t %.3e" %
          (2**(i+3), e1[i], e2[i], e3[i], e4[i]))
print("scipy 2-norm relative error")
for i in range(len(e1)):
    print("%i^2:\t %.3e,\t %.3e,\t %.3e,\t %.3e" %
          (2**(i+3), E1[i], E2[i], E3[i], E4[i]))

print("convergence rate")
for i in range(1, len(e1)):
    n = 2**(i+3)
    r1 = np.log(E1[i]/E1[i-1])/np.log(0.5)
    r2 = np.log(E2[i]/E2[i-1])/np.log(0.5)
    r3 = np.log(E3[i]/E3[i-1])/np.log(0.5)
    r4 = np.log(E4[i]/E4[i-1])/np.log(0.5)
    print("%i^2:\t %.3e,\t %.3e,\t %.3e,\t %.3e" %
          (2**(i+3), r1, r2, r3, r4))
