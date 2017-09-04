# A priori and a posteriori errors

from dolfin import *
import mshr
import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
# set_log_level(1)


# Define Dirichlet boundary (x = 0 or x = 1)
# def BoundLeft(x):
#     return x[0] < DOLFIN_EPS
# def BoundRight(x):
#     return x[0] > 1.0 - DOLFIN_EPS
# def BoundTop(x):
#     return x[1] > 1.0 - DOLFIN_EPS
# def BoundBottom(x):
#     return x[1] < DOLFIN_EPS
def Boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] > 1.0 - \
            DOLFIN_EPS or x[1] < DOLFIN_EPS


def solve_bvp(mesh, f, deg=1):
    ''' define subdomains va Expression() and interpolate to DG-0 elements '''

    V = FunctionSpace(mesh, "CG", deg)
    dx = Measure('dx')

    # bcs = [DirichletBC(V, Constant(0.0), BoundLeft),
    #        DirichletBC(V, Constant(0.0), BoundRight)]
    bcs = DirichletBC(V, Constant(0.0), Boundary)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
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


def error_estimator(mesh, u, f):
    h = CellSize(mesh)
    n = FacetNormal(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    w = TestFunction(DG0)
    # use w as helper function to create elementwise distributed function
    R = h**2*w*(div(grad(u))-f)**2*dx + avg(w)*avg(h)*jump(grad(u), n)**2*dS
    eta_Tsq = Function(DG0)
    assemble(R, tensor=eta_Tsq.vector())
    # global estimator:
    # Ainsworth, Oden 2000
    eta = np.sqrt(eta_Tsq.vector().sum()) #/mesh.num_cells())

    # aI = assemble(h**2*(div(grad(u))-f)**2*dx)
    # aE = assemble(avg(h)*jump(grad(u), n)**2*dS)
    # eta = np.sqrt(aI + aE)
    # # carstensen, Merdon 2010
    # # eta = np.sqrt(aI) + np.sqrt(aE)

    return eta_Tsq, eta


if __name__ == "__main__":

    domain = mshr.Rectangle(Point(0., 0.), Point(1., 1.))

    # f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")

    # adaptive refinement example CC-MC-2010
    # f = Expression("-exp(-100*pow(x[0]-0.5, 2) - pow(x[1] - 117, 2)/10000)*( \
    #                2*(-99 - 4500*x[0] + 24500*x[0]*x[0] - 40000*pow(x[0],3) + \
    #                20000*pow(x[0],4))*x[1]*(x[1]-1) + \
    #                (48830000 + 2341311*x[1] - 11077*x[1]*x[1] - \
    #                235*pow(x[1],3) + pow(x[1],4))/25000000*x[0]*(x[0]-1))")
    # u_ex = Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)*exp(-100*pow(x[0]-0.5, 2) - \
    #                   pow(x[1] - 117, 2)/10000)")

    # f = Expression("pi*pi*(sin(pi*x[0])*sin(pi*x[1]) + \
    #                sin(pi*x[0])*sin(pi*x[1]))")
    # u_ex = Expression("sin(pi*x[0])*sin(pi*x[1])")

# generic
    a = 10
    b = 5
    u_ex = Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)*a*exp(-b*(pow(x[0]-0.5, 2)\
                      + pow(x[1]-0.5, 2)))", a=a, b=b)
    f = Expression("-a*exp(-b*(pow(x[0]-0.5, 2)+pow(x[1]-0.5, 2)))*(\
                   (x[1]-1)*x[1]*(b*b*(x[0]-1)*x[0]*pow(2*x[0]-1,2) -\
                   2*b*(5*x[0]*x[0]-5*x[0]+1)+2) + \
                   (x[0]-1)*x[0]*(b*b*(x[1]-1)*x[1]*pow(2*x[1]-1,2) -\
                   2*b*(5*x[1]*x[1]-5*x[1]+1)+2))", a=a, b=b)


    # NOTE: roc indep. of degree?
    deg = 1

    eta_T = []
    eta = []
    h = []
    errnorm = []
    energynorm = []
    for N in range(3, 9):
        # tic()
        # mesh = mshr.generate_mesh(domain, 2**N)
        # print("built mesh after %.3fs" % toc())
        mesh = UnitSquareMesh(2**N, 2**N)

        u = solve_bvp(mesh, f, deg)

        V = FunctionSpace(mesh, "CG", deg)
        errnorm.append(errornorm(u_ex, u))
        ue = interpolate(u_ex, V)

        eh = project(ue-u, V)  # or eh = ue - interpolate(u, V) ?
        e_sq = sqrt(assemble(dot(grad(eh), grad(eh))*dx(mesh)))
        # which is similar to:
        e_h1 = norm(eh, "H1")
        energynorm.append(e_h1)

        Rsq, Rg = error_estimator(mesh, u, f)
        eta_T.append(np.sqrt(Rsq.vector().array()).max())
        eta.append(Rg)
        # eta.append(np.sqrt(T+E))

        h.append(assemble(CellSize(mesh)*dx))

    print("global a posteriori estimator")
    print("N\t eta \t\t |||e||| \t max(eta_T) \t ||e||")
    for i in range(len(eta)):
        n = (i+3)
        print("%i \t %.3e\t %.3e\t %.3e\t %.3e" %
              (2**n, eta[i], energynorm[i], eta_T[i], errnorm[i]))

    print("rate of convergence")
    for i in range(1, len(eta)):
        n = (i+3)
        rh = np.log(h[i]/h[i-1])

        r1 = np.log(eta[i]/eta[i-1])/rh
        r2 = np.log(energynorm[i]/energynorm[i-1])/rh
        r3 = np.log(eta_T[i]/eta_T[i-1])/rh
        r4 = np.log(errnorm[i]/errnorm[i-1])/rh
        print("%i^2:\t %.2f,\t %.2f,\t %.2f,\t %.2f" % (2**n, r1, r2, r3, r4))

    # plt.figure()
    # plt.ion()
    # plt.loglog(1./np.array(h), rnorm.T, '-o')
    # plt.grid()
    # plt.axis('equal')
