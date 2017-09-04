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
plot(u, title="u")
# # plot(grad(u), title="grad(u)")
interactive()
