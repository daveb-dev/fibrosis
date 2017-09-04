###############################################################################
# 2D Poisson problem in a perforated material (perfectly isolated fibers)
# stationary diffusion/conduction
# -div(A grad(u)) = f
# top/bottom: grad(u).n = 0
# left: u = 0, right: u = 1
####
# outline
#   - define functions, classes
#   - solve cell problems -div(a grad(w_k)) = div(a e_k),  k=1,2
#       with a = const scalar
#       on Y = (0,1)^2 unit square with circular perforation, R=0.25
#       periodicity on the outer boundaries
#       Neumann BC on the circle: a (grad(w_k)+e_k).n = 0
#                            <=>        a grad(w_k).n = -a e_k.n
#       constraint <w_k> = int_Y w_k dy = 0
#   - compute effective diffusivity a*
#   - solve homogenized problem -div(a* grad(u0)) = f  w/ BCs
#   - calculate corrector u1 = u0 + w_k du0/dx_k

# singular problem/constraint:
# http://fenicsproject.org/documentation/dolfin/dev/python/demo/documented/neumann-poisson/python/documentation.html
# http://fenicsproject.org/documentation/dolfin/dev/python/demo/documented/singular-poisson/python/documentation.html
# http://fenicsproject.org/qa/2406/solve-poisson-problem-with-neumann-bc

# TODO: corrector not implemented, cf.
# http://fenicsproject.org/qa/8628/interpolate-cell-solution-periodically-to-a-bigger-mesh
#

import numpy as np
from dolfin import *
import mshr

set_log_level(1)


# DEFINE CELL PROBLEM
class CircleBoundary(SubDomain):
    def inside(self, x, on_boundary):
        # return points that lie on circle w/ center (X0/Y0) and radius R+eps
        # Center coordinates
        X0, Y0 = 0.5, 0.5  # 0.5, 0.5
        R = 0.25
        return bool(sqrt((x[0]-X0)**2 + (x[1]-Y0)**2) < R + DOLFIN_EPS and
                    on_boundary)


class PeriodicBoundary(SubDomain):
    # Left/bottom boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two
        # corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], 1)) or
                          (near(x[0], 1) and near(x[1], 0)))) and on_boundary)
        # return bool((near(x[0], -0.5) or near(x[1], -0.5)) and
        #             (not ((near(x[0], -0.5) and near(x[1], 0.5)) or
        #                   (near(x[0], 0.5) and near(x[1], -0.5))))
        #           and on_boundary)

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

# unit cell geometry
# 1) diameter of perforation = 1/2 cell length
R = 0.25
# 2) circle area = 1/2 square area
# R = np.sqrt(0.5/np.pi)
# domain: unit square minus circle with radius R
domain = mshr.Rectangle(Point(0., 0.), Point(1., 1.)) - \
        mshr.Circle(Point(.5, .5), R)
# domain = mshr.Rectangle(Point(-0.5, -0.5), Point(0.5, 0.5)) - \
#         mshr.Circle(Point(0., 0.), R)
# generate mesh with ~32 elements/axis
mesh = mshr.generate_mesh(domain, 2**7)

# mark boundaries
# init and mark circle for Neumann boundary as "1"
circle = CircleBoundary()
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
circle.mark(boundaries, 1)

# def boundary measure ds. ds(1) circle, ds(0) all others
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# weak formulation
# periodic function space V "Continuous Galerkin" with standard Lagrange
# elements
V = FunctionSpace(mesh, "DG", 1, constrained_domain=PeriodicBoundary())

# define unit vectors e_k
e1 = Constant([1., 0.])
e2 = Constant([0., 1.])

# physical parameters
Ay = Constant(1.0*np.eye(2))

# USE TWO METHODS: KRYLOV SPACE (1) vs LAGRANGE-MULTIPLIER (2)
# (0) --> ignore
method = 1
if method == 1:  # KRYLOV
    # get facet normals of whole mesh
    n = FacetNormal(mesh)

    u = TrialFunction(V)
    v = TestFunction(V)

# k == 1
# build Neumann condition for RHS boundary integral for k=1 (w1,e1)
    g = -dot(dot(Ay, e1), n)*v*ds(1)
# build RHS "source" term
    f = -dot(dot(Ay, e1), grad(v))*dx
# bilinear+linear forms
    a = inner(dot(Ay, grad(u)), grad(v))*dx
    L = f + g
# TODO: INCLUDING f CHANGES RESULTS. WHY?
# TODO: CHECK SIGNS in L !!

    A = assemble(a)
    b = assemble(L)
    e = Function(V)
    e.interpolate(Constant(1.0))
    evec = e.vector()
    evec /= norm(evec)
    alpha = b.inner(evec)
    b -= alpha*evec

    x1 = Function(V)

    prec = PETScPreconditioner('hypre_amg')
    PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
    solver = PETScKrylovSolver('cg', prec)
    solver.parameters['absolute_tolerance'] = 0.0
    solver.parameters['relative_tolerance'] = 1.0e-10
    solver.parameters['maximum_iterations'] = 100
    solver.parameters['monitor_convergence'] = True
# Create solver and solve system
    A_petsc = as_backend_type(A)
    b_petsc = as_backend_type(b)
    x_petsc = as_backend_type(x1.vector())
    solver.set_operator(A_petsc)
    solver.solve(x_petsc, b_petsc)

    w1 = x1


# check if constraint <w_1> = 0 is fulfilled
# integrate solution over domain
    X1 = assemble(w1*dx)

#  k == 2
# build Neumann condition for RHS boundary integral for k=1 (w1,e1)
    g = -dot(dot(Ay, e2), n)*v*ds(1)
    f = -dot(dot(Ay, e2), grad(v))*dx
# bilinear form and A matrix unchanged
# linear form:
    L = f + g

    b = assemble(L)
    alpha = b.inner(evec)
    b -= alpha*evec

    x2 = Function(V)

    prec = PETScPreconditioner('hypre_amg')
    PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
    solver = PETScKrylovSolver('cg', prec)
    solver.parameters['absolute_tolerance'] = 0.0
    solver.parameters['relative_tolerance'] = 1.0e-10
    solver.parameters['maximum_iterations'] = 100
    solver.parameters['monitor_convergence'] = True
# Create solver and solve system
    A_petsc = as_backend_type(A)
    b_petsc = as_backend_type(b)
    x_petsc = as_backend_type(x2.vector())
    solver.set_operator(A_petsc)
    solver.solve(x_petsc, b_petsc)

    w2 = x2

# check if constraint <w_1> = 0 is fulfilled
# integrate solution over domain
    X2 = assemble(w2*dx)
    print("check constraint <w1> = %f" % X1)
    print("check constraint <w2> = %f" % X2)

elif method == 2:       # MIXED SPACE; LAGRANGE MULTIPLIER
    # Real, where constant "c" is searched for (-> constraint)
    R = FunctionSpace(mesh, "R", 0, constrained_domain=PeriodicBoundary())
# Mixed function space
    W = V * R

# variational problem
    (u, c) = TrialFunction(W)
    (v, d) = TestFunction(W)

    n = FacetNormal(mesh)

#  k == 1
# build Neumann condition for RHS boundary integral for k=1 (w1,e1)
    g = -dot(dot(Ay, e1), n)

    f = -dot(dot(Ay, e1), grad(v))*dx
# bilinear+linear forms
    a = (inner(dot(Ay, grad(u)), grad(v)) + c*v + u*d)*dx
    L = f + g*v*ds(1)

# solve
    w = Function(W)
    solve(a == L, w)
    (w1, c1) = w.split(deepcopy=True)

# check constraint. integrate w2 over domain
    X1 = assemble(w1*dx)

#  k == 2

    f = -dot(dot(Ay, e2), grad(v))*dx
    g = -dot(dot(Ay, e2), n)
    L = f + g*v*ds(1)

# solve
    solve(a == L, w)
    (w2, c2) = w.split(deepcopy=True)
    X2 = assemble(w2*dx)
    print("check constraint <w1> = %f" % X1)
    print("check constraint <w2> = %f" % X2)

else:  # none
    # get facet normals of whole mesh
    n = FacetNormal(mesh)

    u = TrialFunction(V)
    v = TestFunction(V)

# build Neumann condition for RHS boundary integral for k=1 (w1,e1)
    g = -dot(dot(Ay, e1), n)

    f = dot(dot(Ay, e1), grad(v))*dx
# bilinear+linear forms
    a = inner(dot(Ay, grad(u)), grad(v))*dx
    L = g*v*ds(1)

# solve
    w1 = Function(V)
    solve(a == L, w1)


# Compute cell solution w2 for e2
    g = -dot(dot(Ay, e2), n)
    L = g*v*ds(1)

    w2 = Function(V)
    solve(a == L, w2)


# Plot solution
# plot(w1, interactive=True)
# plot(w2, interactive=True)

# EFFECTIVE COEFFICIENT
# # compute gradient of w_k via projection
# Vg = VectorFunctionSpace(mesh, 'CG', 1)
# R = VectorFunctionSpace(mesh, 'R', 0)  # real "placeholder"
# c = TestFunction(R)

# grad_w1 = project(grad(w1), Vg)
# grad_w2 = project(grad(w2), Vg)
# # dw1x, dw1y = grad_w1.split(deepcopy=True)
# # dw2x, dw2y = grad_w2.split(deepcopy=True)

# A1 = dot(dot(Ay, -grad_w1+e1), c)*dx
# A2 = dot(dot(Ay, -grad_w2+e2), c)*dx

# Aeff = np.eye(2)
# Aeff[:, 0] = assemble(A1)
# Aeff[:, 1] = assemble(A2)

# Ae = Constant(Aeff)

# # TODO: watch out: domain sizes different (Omega_eps vs Omega)

# # HOMOGENEOUS PROBLEM

# domain_h = mshr.Rectangle(Point(0., 0.), Point(1., 1.))
# # generate mesh with ~32 elements/axis
# mesh_h = mshr.generate_mesh(domain_h, 2**6)


# class Left(SubDomain):
#     def inside(self, x, on_boundary):
#         return near(x[0], 0.0)


# class Right(SubDomain):
#     def inside(self, x, on_boundary):
#         return near(x[0], 1.0)

# # Initialize sub-domain instances
# left = Left()
# right = Right()

# # Initialize mesh function for boundary domains
# boundaries = FacetFunction("size_t", mesh_h)
# boundaries.set_all(0)
# left.mark(boundaries, 1)
# right.mark(boundaries, 2)

# f = Constant(1.0)

# # Define function space and basis functions
# Vh = FunctionSpace(mesh_h, "CG", 1)
# u = TrialFunction(Vh)
# v = TestFunction(Vh)

# # Define Dirichlet boundary conditions at top and bottom boundaries
# bcs = [DirichletBC(Vh, 1.0, boundaries, 2),
#        DirichletBC(Vh, 0.0, boundaries, 1)]

# f = Constant(.0)

# a = inner(dot(Ae, grad(u)), grad(v))*dx
# L = f*v*dx

# u0 = Function(Vh)
# solve(a == L, u0, bcs)


# # CORRECTOR 1st order
# # w = as_vector([w1, w2])
# grad_u0 = project(grad(u0), VectorFunctionSpace(mesh, "CG", 1))
# du0x1, du0x2 = grad_u0.split(deepcopy=True)

# # TODO does this make sense!?? x, y=x/eps independet, separated variables
# u1 = du0x1*w1 + du0x2*w2    # dot(grad_u0, w)
# # here domains of wk and u0 equal (incidentally) -> multiplication of Y cell
# # with whole domain

# # TODO: u = u0 + eps*u1
# #       i.e., 'shrink' u1 to eps-mesh and copy over domain, interpolate??
# # NOTE!     coordinates not included in solution


# # Plot solutions
# # plot(u0, title="u_0")
# # plot(u1, title="u_1")
# # interactive()
