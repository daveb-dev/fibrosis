from dolfin import *


def tube_mesh2D(R, L, N):
    import mshr
    domain = mshr.Rectangle(Point(0.0, -R), Point(L, R))
    mesh = mshr.generate_mesh(domain, N)
    return mesh


def pipe_mesh3D(R, L, N):
    import mshr
    domain = mshr.Cylinder(Point(0, 0, 0), Point(L, 0, 0), R, R)
    mesh = mshr.generate_mesh(domain, N)
    return mesh


# backward facing step
def BFSmesh(H, dH, L1, L2, N):
    import mshr
    domain = mshr.Rectangle(Point(0.0, dH), Point(L1, H)) + \
        mshr.Rectangle(Point(L1, 0.0), Point(L1+L2, H))
    mesh = mshr.generate_mesh(domain, N)
    mesh._name = 'BFS'
    return mesh


# Channel flow around cylinder
def cylinderflow_mesh(W=1, L=5, Lc=1, Rc=0.18, N=32):
    import mshr
    domain = mshr.Rectangle(Point(0.0, -W/2.), Point(L, W/2.)) - \
        mshr.Circle(Point(Lc, 0.0), Rc)
    mesh = mshr.generate_mesh(domain, N)
    mesh._name = 'cyl'
    return mesh


# Sub domain for Periodic boundary condition at inlet
class PeriodicBoundaryINLET(SubDomain):
    def __init__(self, y):
        self.dx = y
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(x[0] < self.dx and x[1] < DOLFIN_EPS and
                    x[1] > -DOLFIN_EPS and on_boundary)

    # Map top boundary (H) to bottom boundary (G)
    def map(self, x, y):
        y[1] = x[1] - 1.0
        y[0] = x[0]


class AllBoundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class Left(SubDomain):
    def __init__(self, y):
        self.L = y
        SubDomain.__init__(self)  # Call base class constructor!

    def inside(self, x, on_boundary):
        return x[0] < self.L + DOLFIN_EPS and on_boundary


class Right(SubDomain):
    def __init__(self, y):
        self.L = y
        SubDomain.__init__(self)  # Call base class constructor!

    def inside(self, x, on_boundary):
        tol = 1e-14
        return x[0] > self.L - tol and on_boundary  # DOLFIN_EPS


class Top(SubDomain):
    def __init__(self, y):
        self.L = y
        SubDomain.__init__(self)  # Call base class constructor!

    def inside(self, x, on_boundary):
        tol = 1e-14
        return x[1] > self.L - tol and on_boundary  # DOLFIN_EPS


class Bottom(SubDomain):
    def __init__(self, y):
        self.L = y
        SubDomain.__init__(self)  # Call base class constructor!

    def inside(self, x, on_boundary):
        return x[1] < self.L + DOLFIN_EPS and on_boundary


class Corner(SubDomain):
    ''' Right lower corner of the mesh '''
    def __init__(self, xp):
        self.xp = xp
        SubDomain.__init__(self)  # Call base class constructor!

    def inside(self, x, on_boundary):
        # don't require "on_boundary" here!

        ret = (near(x[0], self.xp[0]) and near(x[1], self.xp[1]) and
               (near(x[2], self.xp[2]) if len(self.xp) == 3 else True))

        return ret
