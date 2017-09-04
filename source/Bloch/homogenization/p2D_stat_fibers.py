###############################################################################
# 2D Poisson problem in a composite material (matrix w/ conducting fibers)
# stationary diffusion/conduction
# -div(A grad(u)) = f
# top/bottom: grad(u).n = 0
# left: u = 0, right: u = 1
# A = Af in fiber, Am in matrix
####
# outline
#   - define functions, classes
#   - solve cell problems -div(A grad(w_k)) = div(A e_k),  k=1,2
#       with a = const scalar
#       on Y = (0,1)^2 unit square with circular R=0.25 subdomain
#       periodicity on the outer boundaries
#       interface continuity condition  on the circle:
#                            <=>       TODO
#       constraint <w_k> = int_Y w_k dy = 0
#   - compute effective diffusivity A*
#   - solve homogenized problem -div(A* grad(u0)) = f  w/ BCs
#   - calculate corrector u1 = u0 + w_k du0/dx_k

# singular problem/constraint:
# http://fenicsproject.org/documentation/dolfin/dev/python/demo/documented/neumann-poisson/python/documentation.html
# http://fenicsproject.org/documentation/dolfin/dev/python/demo/documented/singular-poisson/python/documentation.html
# http://fenicsproject.org/qa/2406/solve-poisson-problem-with-neumann-bc


import numpy as np
from dolfin import *
import mshr

set_log_level(1)


# DEFINE CELL PROBLEM
def CellProblem(Am=1.0, Af=1.1, Nx=128):
    class Circle(SubDomain):    # not used currently
        def inside(self, x, on_boundary):
            # return points that lie within circle
            # w/ center (Xc,Yc) and radius R+eps
            Xc, Yc = 0.5, 0.5
            R = 0.25
            return bool(sqrt((x[0]-Xc)**2 + (x[1]-Yc)**2) < R + DOLFIN_EPS)

    class PeriodicBoundary(SubDomain):
        # Left/bottom boundary is "target domain" G
        def inside(self, x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the
            # two corners (0, 1) and (1, 0)
            return bool((near(x[0], 0) or near(x[1], 0)) and
                        (not ((near(x[0], 0) and near(x[1], 1)) or
                              (near(x[0], 1) and near(x[1], 0)))) and
                        on_boundary)

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

    # Unit cell
    domain = mshr.Rectangle(Point(0., 0.), Point(1., 1.))
    mesh = mshr.generate_mesh(domain, Nx)

    Am = Constant(Am)
    Af = Constant(Af)

# Initialize mesh function for interior domains
# domains = CellFunction("size_t", mesh)
# domains.set_all(0)
# # Mark circle subdomain as "1"
# circle = Circle()
# circle.mark(domains, 1)

# def boundary measure ds. dx(1) circle/fiber, dx(0) rest/matrix
    dx = Measure('dx')

# weak formulation
# periodic function space V "Continuous Galerkin" with standard Lagrange
# elements
    V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())
    V0 = FunctionSpace(mesh, "DG", 0)

# define unit vectors e_k
    e1 = Constant([1., 0.])
    e2 = Constant([0., 1.])

    Afun = Expression('sqrt(pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2)) \
                      < 0.25 + DOLFIN_EPS ?  Af : Am',
                      Af=float(Af), Am=float(Am))
# USE TWO METHODS: KRYLOV SPACE (1) vs LAGRANGE-MULTIPLIER (2)
# (0) --> ignore
    method = 1
    if method == 1:  # KRYLOV
        # get facet normals of whole mesh
        u = TrialFunction(V)
        v = TestFunction(V)

# k == 1

        Ai = interpolate(Afun, V0)

        a = inner(Ai*grad(u), grad(v))*dx
        L = -Ai*dot(e1, grad(v))*dx

        # domains = CellFunction("size_t", mesh)
        # domains.set_all(0)
# # Mark circle subdomain as "1"
        # circle = Circle()
        # circle.mark(domains, 1)

# # def boundary measure ds. dx(1) circle/fiber, dx(0) rest/matrix
        # dx = Measure('dx')[domains]
# # Bilinear+linear forms
        # a = inner(Am*grad(u), grad(v))*dx(0) + inner(Af*grad(u), grad(v))*dx(1)
        # L = -(dot(Am*e1, grad(v))*dx(0) + dot(Af*e1, grad(v))*dx(1))

# Fix Krylov Solver for singular problem
        A = assemble(a)
        b = assemble(L)
        e = Function(V)
        e.interpolate(Constant(1.0))
        evec = e.vector()
        evec /= norm(evec)
        alpha = b.inner(evec)
        b -= alpha*evec

        x = Function(V)

        prec = PETScPreconditioner('hypre_amg')   # hypre_amg, ilu
        PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
        solver = PETScKrylovSolver('cg', prec)
        solver.parameters['absolute_tolerance'] = 0.0
        solver.parameters['relative_tolerance'] = 1.0e-10
        solver.parameters['maximum_iterations'] = 200
        solver.parameters['monitor_convergence'] = True
# Create solver and solve system
        A_petsc = as_backend_type(A)
        b_petsc = as_backend_type(b)
        x_petsc = as_backend_type(x.vector())
        solver.set_operator(A_petsc)
        solver.solve(x_petsc, b_petsc)

        w1 = x


# check if constraint <w_1> = 0 is fulfilled
# integrate solution over domain
        X1 = assemble(w1*dx)

#  k == 2
        L = -Ai*dot(e2, grad(v))*dx
        # L = -Am*dot(e2, grad(v))*dx(0) - Af*dot(e2, grad(v))*dx(1)

        b = assemble(L)
        alpha = b.inner(evec)
        b -= alpha*evec

        x = Function(V)

# Create solver and solve system
        A_petsc = as_backend_type(A)
        b_petsc = as_backend_type(b)
        x_petsc = as_backend_type(x.vector())
        solver.set_operator(A_petsc)
        solver.solve(x_petsc, b_petsc)

        w2 = x
# check if constraint <w_1> = 0 is fulfilled
# integrate solution over domain
        print("check constraint <w1> = %f" % X1)

    elif method == 2:       # MIXED SPACE; LAGRANGE MULTIPLIER
        # Real, where constant "c" is searched for (-> constraint)
        R = FunctionSpace(mesh, "R", 0, constrained_domain=PeriodicBoundary())
# Mixed function space
        W = V * R

# variational problem
        (u, c) = TrialFunction(W)
        (v, d) = TestFunction(W)

#  k == 1

        domains = CellFunction("size_t", mesh)
        domains.set_all(0)
# Mark circle subdomain as "1"
        circle = Circle()
        circle.mark(domains, 1)

# def boundary measure ds. dx(1) circle/fiber, dx(0) rest/matrix
        dx = Measure('dx')[domains]
# Bilinear+linear forms
        a = inner(Am*grad(u), grad(v))*dx(0) + inner(Af*grad(u), grad(v))*dx(1)
        L = -Am*dot(e1, grad(v))*dx(0) - Af*dot(e1, grad(v))*dx(1)

# solve
        w = Function(W)
        solve(a == L, w)
        (w1, c1) = w.split(deepcopy=True)

# check constraint. integrate w2 over domain
        X1 = assemble(w1*dx)

#  k == 2
        L = -Am*dot(e2, grad(v))*dx(0) - Af*dot(e2, grad(v))*dx(1)
        w = Function(W)
        solve(a == L, w)
        (w2, c2) = w.split(deepcopy=True)
        print("check constraint <w1> = %f" % X1)

    else:  # none
        print('choose a side...')

    return w1, w2

w1, w2 = CellProblem()
# Plot solution
# plot(w1, interactive=True)
# plot(w2, interactive=True)

# # EFFECTIVE COEFFICIENT
# # compute gradient of w_k via projection
# Vg = VectorFunctionSpace(mesh, 'CG', 1)
# R = VectorFunctionSpace(mesh, 'R', 0)  # real "placeholder"
# c = TestFunction(R)

# grad_w1 = project(grad(w1), Vg)
# # grad_w2 = project(grad(w2), Vg)
# # dw1x, dw1y = grad_w1.split(deepcopy=True)
# # dw2x, dw2y = grad_w2.split(deepcopy=True)

# # # TODO stupid hack, can be done better (I guess)
# # A = Am*dot(-grad_w1+e1, c)*dx(0) + Af*dot(-grad_w1+e1, c)*dx(1)
# # # A2 = dot(dot(Ay, -grad_w2+e2), c)*dx

# # Aeff = np.eye(2)
# # Aeff[:, 0] = assemble(A)
# # Aeff[:, 1] = assemble(A)

# # Ae = Constant(Aeff)

# # # TODO: watch out: domain sizes different (Omega_eps vs Omega)

# # # HOMOGENEOUS PROBLEM

# # domain_h = mshr.Rectangle(Point(0., 0.), Point(1., 1.))
# # # generate mesh with ~32 elements/axis
# # mesh_h = mshr.generate_mesh(domain_h, 2**6)


# # class Left(SubDomain):
# #     def inside(self, x, on_boundary):
# #         return near(x[0], 0.0)


# # class Right(SubDomain):
# #     def inside(self, x, on_boundary):
# #         return near(x[0], 1.0)

# # # Initialize sub-domain instances
# # left = Left()
# # right = Right()

# # # Initialize mesh function for boundary domains
# # boundaries = FacetFunction("size_t", mesh_h)
# # boundaries.set_all(0)
# # left.mark(boundaries, 1)
# # right.mark(boundaries, 2)

# # domains_h = CellFunction("size_t", mesh_h)
# # domains_h.set_all(0)

# # dx = Measure("dx")[domains_h]

# # # Define function space and basis functions
# # Vh = FunctionSpace(mesh_h, "CG", 1)
# # u = TrialFunction(Vh)
# # v = TestFunction(Vh)

# # # Define Dirichlet boundary conditions at top and bottom boundaries
# # bcs = [DirichletBC(Vh, 1.0, boundaries, 2),
# #        DirichletBC(Vh, 0.0, boundaries, 1)]

# # f = Constant(.0)

# # a = inner(dot(Ae, grad(u)), grad(v))*dx
# # L = f*v*dx

# # u0 = Function(Vh)
# # solve(a == L, u0, bcs)


# # # CORRECTOR 1st order
# # # w = as_vector([w1, w2])
# # grad_u0 = project(grad(u0), VectorFunctionSpace(mesh, "CG", 1))
# # du0x1, du0x2 = grad_u0.split(deepcopy=True)

# # # TODO does this make sense!?? x, y=x/eps independet, separated variables
# # u1 = du0x1*w1 + du0x2*w2    # dot(grad_u0, w)
# # # here domains of wk and u0 equal (incidentally) -> multiplication of Y cell
# # # with whole domain

# # # TODO: u = u0 + eps*u1
# # #       i.e., 'shrink' u1 to eps-mesh and copy over domain, interpolate??
# # # NOTE!     coordinates not included in solution


# # # Plot solutions
# # # plot(u0, title="u_0")
# # # plot(u1, title="u_1")
# # # interactive()
