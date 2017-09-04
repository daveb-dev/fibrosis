# 2D Poisson problem in perforated material (perfectly isolated fibers)
# full model without homogenization

import numpy as np
from dolfin import *
import mshr


## Geometry & Mesh
# number of cells/direction
Ncell = 10
# domain length
L = 1.
eps = L/Ncell
# perforation radius in unit cell
R = 0.25
# alternative: circle area = 1/2 square area
# R = np.sqrt(0.5/np.pi)

domain = mshr.Rectangle(Point(0., 0.), Point(1., 1.))
for i in range(Ncell):
    for j in range(Ncell):
        domain = domain - mshr.Circle(Point((i+0.5)*eps,(j+0.5)*eps), R*eps)

mesh = mshr.generate_mesh(domain, Ncell*16)

# plot(mesh, interactive=True)

## Boundary conditions 

# Left and right -> Dirichlet
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)



# class PeriodicBoundary(SubDomain):
#     # Left/Bottom boundary is "target domain" G
#     def inside(self, x, on_boundary):
#         return bool((x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary))
#         #        or (x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS and on_boundary))

#     # Map right boundary (H) to left boundary (G)
#     def map(self, x, y):
#         y[0] = x[0] - 1.0
#         y[1] = x[1] #- 1.0

# Source term
class Source(Expression):
    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] = x[0]*sin(5.0*DOLFIN_PI*x[1]) \
                    + 1.0*exp(-(dx*dx + dy*dy)/0.02)
# Initialize sub-domain instances
left = Left()
right = Right()

# Initialize mesh function for boundary domains
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)

# Define input data
a0 = Constant(1.0)
f = Constant(.0)

# Define function space and basis functions
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

# Define Dirichlet boundary conditions at top and bottom boundaries
bcs = [DirichletBC(V, 1.0, boundaries, 2),
       DirichletBC(V, 0.0, boundaries, 1)]

# Neumann on Circles and top/bottom boundary

a = inner(a0*grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# Plot solution
# plot(u, title="u")
# # plot(grad(u), title="grad(u)")
# interactive()

## HOMOGENEOUS PROBLEM w/o HOLES

domain = mshr.Rectangle(Point(0., 0.), Point(1., 1.))
mesh = mshr.generate_mesh(domain, Ncell*16)

V2 = FunctionSpace(mesh, "CG", 1)
u2 = TrialFunction(V2)
v2 = TestFunction(V2)

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
bcs = [DirichletBC(V2, 1.0, boundaries, 2),
       DirichletBC(V2, 0.0, boundaries, 1)]

a = inner(a0*grad(u2), grad(v2))*dx
L = f*v2*dx

# Compute solution
u2 = Function(V2)
solve(a == L, u2, bcs)

dif = Function(V)
dif.vector()[:] = u.vector() - interpolate(u2, V).vector()

plot(u, title="perforated domain")
plot(u2, title="without holes")
plot(dif, title="difference u_perf - u_hom")
interactive()


