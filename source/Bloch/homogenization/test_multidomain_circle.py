# compare Subdomains vs Expression

from dolfin import *
import mshr
import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
# set_log_level(1)


# DEFINE CELL PROBLEM
class Circle(SubDomain):
    def inside(self, x, on_boundary):
        # return points that lie within circle
        # w/ center (0.5,0.5) and R=0.25+eps
        return bool(sqrt((x[0]-0.5)**2 + (x[1]-0.5)**2) <= 0.25)


class PeriodicBoundary(SubDomain):
    # Left/bottom boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two
        # corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], 1)) or
                          (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.


# Define Dirichlet boundary (x = 0 or x = 1)
def BoundLeft(x):
    return x[0] < DOLFIN_EPS


def BoundRight(x):
    return x[0] > 1.0 - DOLFIN_EPS


def solve_subdomains(mesh, deg=1, f=0):
    ''' use subdomains to handle discontinuity '''

    V = FunctionSpace(mesh, "CG", deg)

    domains = CellFunction("size_t", mesh)
    domains.set_all(0)
    circle = Circle()
    circle.mark(domains, 1)

    dx = Measure('dx')[domains]

    kf = 2.0
    km = 1.0

    if not type(f) == 'dolfin.functions.constant.Constant':
        f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) \
                       / 0.02)")

    bcs = [DirichletBC(V, Constant(0.0), BoundLeft),
           DirichletBC(V, Constant(0.0), BoundRight)]

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(km*grad(u), grad(v))*dx(0) + inner(kf*grad(u), grad(v))*dx(1)
    L = f*v*dx

    tic()
    A, b = assemble_system(a, L, bcs)
    u = Function(V)
    t_asmbl = toc()

    tic()
    parameters["linear_algebra_backend"] = "PETSc"  # alread set (default)?

    prec = PETScPreconditioner('hypre_amg')   # hypre_amg, ilu
    PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
    solver = PETScKrylovSolver('cg', prec)
    # solver = PETScKrylovSolver('cg')
    solver.parameters['absolute_tolerance'] = 0.0
    solver.parameters['relative_tolerance'] = 1.0e-8
    solver.parameters['maximum_iterations'] = 100
    solver.parameters['monitor_convergence'] = True
    # Create solver and solve system
    A_petsc = as_backend_type(A)
    b_petsc = as_backend_type(b)
    x_petsc = as_backend_type(u.vector())
    solver.set_operator(A_petsc)
    solver.solve(x_petsc, b_petsc)
    t_solve = toc()

    # u = Function(V)
    # tic()
    # solve(a == L, u, bcs)
    # t_solve = toc()

    print("total: %.3fs" % (t_asmbl+t_solve))
    return u


def solve_interp(mesh, deg=1, f=0):
    ''' define subdomains va Expression() and interpolate to DG-0 elements '''

    V = FunctionSpace(mesh, "CG", deg)
    dx = Measure('dx')

    bcs = [DirichletBC(V, Constant(0.0), BoundLeft),
           DirichletBC(V, Constant(0.0), BoundRight)]

    kf = 2.0
    km = 1.0
    k_ex = Expression('sqrt(pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2)) \
                <= 0.25 ?  kf : km', kf=kf, km=km)
    if not type(f) == 'dolfin.functions.constant.Constant':
        f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) \
                       / 0.02)")

    u = TrialFunction(V)
    v = TestFunction(V)
    V0 = FunctionSpace(mesh, "DG", 0)  # Discontinuous Galerkin
    k = interpolate(k_ex, V0)
    a = inner(k*grad(u), grad(v))*dx
    L = f*v*dx

    tic()
    A, b = assemble_system(a, L, bcs)
    u = Function(V)
    t_asmbl = toc()

    tic()
    parameters["linear_algebra_backend"] = "PETSc"  # alread set (default)?

    prec = PETScPreconditioner('hypre_amg')   # hypre_amg, ilu
    PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
    solver = PETScKrylovSolver('cg', prec)
    # solver = PETScKrylovSolver('cg')
    solver.parameters['absolute_tolerance'] = 0.0
    solver.parameters['relative_tolerance'] = 1.0e-8
    solver.parameters['maximum_iterations'] = 100
    solver.parameters['monitor_convergence'] = True
    # Create solver and solve system
    A_petsc = as_backend_type(A)
    b_petsc = as_backend_type(b)
    x_petsc = as_backend_type(u.vector())
    solver.set_operator(A_petsc)
    solver.solve(x_petsc, b_petsc)
    t_solve = toc()

    # u = Function(V)
    # tic()
    # solve(a == L, u, bcs)
    # t_solve = toc()

    print("total: %.3fs" % (t_asmbl+t_solve))

    return u


def solve_conditional(mesh, deg=1, f=0):
    ''' define subdomains va Expression() and interpolate to DG-0 elements '''

    V = FunctionSpace(mesh, "CG", deg)
    dx = Measure('dx')

    bcs = [DirichletBC(V, Constant(0.0), BoundLeft),
           DirichletBC(V, Constant(0.0), BoundRight)]

    kf = 2.0
    km = 1.0
    x0, x1 = MeshCoordinates(mesh)
    k = conditional(sqrt((x0-0.5)**2 + (x1-0.5)**2) <= 0.25, kf, km)
    if not type(f) == 'dolfin.functions.constant.Constant':
        f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) \
                       / 0.02)")

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(k*grad(u), grad(v))*dx
    L = f*v*dx

    # build linear system for Krylov Solver
    # A = assemble(a)
    # b = assemble(L)
    # for bc in bcs:
    #     bc.apply(A, b)

    # or in one line:
    tic()
    A, b = assemble_system(a, L, bcs)
    u = Function(V)
    t_asmbl = toc()

    tic()
    parameters["linear_algebra_backend"] = "PETSc"  # alread set (default)?

    prec = PETScPreconditioner('hypre_amg')   # hypre_amg, ilu
    PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
    solver = PETScKrylovSolver('cg', prec)
    # solver = PETScKrylovSolver('cg')
    solver.parameters['absolute_tolerance'] = 0.0
    solver.parameters['relative_tolerance'] = 1.0e-8
    solver.parameters['maximum_iterations'] = 100
    solver.parameters['monitor_convergence'] = True
    # Create solver and solve system
    A_petsc = as_backend_type(A)
    b_petsc = as_backend_type(b)
    x_petsc = as_backend_type(u.vector())
    solver.set_operator(A_petsc)
    solver.solve(x_petsc, b_petsc)
    t_solve = toc()

    # u = Function(V)
    # tic()
    # solve(a == L, u, bcs)
    # t_solve = toc()

    print("assembled system in  %.3fs" % t_asmbl)
    print("solved system in  %.3fs" % t_solve)
    print("total: %.3fs" % (t_asmbl+t_solve))

    # u automatically updated, == x_petsc
    return u


def residuals(mesh, u, f):
    h = CellSize(mesh)
    n = FacetNormal(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    w = TestFunction(DG0)
    # TODO: COEFFICIENT MISSING HERE
    residual = h**2*w*(div(grad(u))-f)**2*dx + avg(w)*avg(h)*jump(grad(u),
                                                                  n)**2*dS
    cell_residual = Function(DG0)
    assemble(residual, tensor=cell_residual.vector())
    return cell_residual


if __name__ == "__main__":

    domain = mshr.Rectangle(Point(0., 0.), Point(1., 1.))

    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
    # f = Expression("pi*pi*pow(sin(pi*x[0]), 2)*pow(sin(pi*x[1]), 2)")

    # NOTE: roc indep. of degree?
    deg = 1

    res1, res2, res3 = [], [], []
    h = []
    for N in range(3, 9):
        # tic()
        # mesh = mshr.generate_mesh(domain, 2**N)
        # print("built mesh after %.3fs" % toc())
        mesh = UnitSquareMesh(2**N, 2**N)

        u1 = solve_subdomains(mesh, deg, f)
        u2 = solve_interp(mesh, deg, f)
        u3 = solve_conditional(mesh, deg, f)

        res1.append(residuals(mesh, u1, f))
        res2.append(residuals(mesh, u2, f))
        res3.append(residuals(mesh, u3, f))

        h.append(assemble(CellSize(mesh)*dx))

    print("residual norms")
    print("N \t[1]  inf     L2 \t[2]  inf     L2 \t\
[3]  inf     L2")
    rnorm = np.zeros((3, len(res1)))
    for i in range(len(res1)):
        n = (i+3)**2
        # L2 norms of functions
        # rnorm[0, i] = norm(res1[i])
        # rnorm[1, i] = norm(res2[i])
        # rnorm[2, i] = norm(res3[i])
        # NOTE: multiply by h or not!?
        rnorm[0, i] = la.norm(np.sqrt(res1[i].vector().array()), 2)*h[i]
        rnorm[1, i] = la.norm(np.sqrt(res2[i].vector().array()), 2)*h[i]
        rnorm[2, i] = la.norm(np.sqrt(res3[i].vector().array()), 2)*h[i]
        print("%i \t %.3e   %.3e\t %.3e   %.3e\t %.3e   %.3e" %
              (n, res1[i].vector().max(), rnorm[0, i],
               res2[i].vector().max(), rnorm[1, i],
               res3[i].vector().max(), rnorm[2, i]))

    print("rate of convergence")
    for i in range(1, len(res1)):
        n = (i+3)**2
        rh = np.log(h[i]/h[i-1])
        r1 = np.log(rnorm[0, i]/rnorm[0, i-1])/rh
        r2 = np.log(rnorm[1, i]/rnorm[1, i-1])/rh
        r3 = np.log(rnorm[2, i]/rnorm[2, i-1])/rh
        print("%i^2:\t %.2f,\t %.2f,\t %.2f" % (n, r1, r2, r3))

    plt.figure()
    plt.ion()
    plt.loglog(1./np.array(h), rnorm.T, '-o')
    plt.grid()
    plt.axis('equal')
