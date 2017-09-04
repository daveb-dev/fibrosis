from dolfin import *
from functions.geom import *
import mshr

parameters["form_compiler"]["precision"] = 100
# stokes error approximately halved w.r.t. value 15
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = \
    "-O3 -ffast-math -march=native"
# very slight effect on precision of stokes (neither pos nor neg)

U = 100.
N = 32
fix_pressure = 1    # 0: ignore, 1: correct -p0, 2: point dBC
meth = 1
mu = Constant(0.035)
rho = Constant(1.0)

geo = mshr.Rectangle(Point(0., -1.), Point(6., 1.))
mesh = mshr.generate_mesh(geo, N)
ndim = mesh.topology().dim()

V = VectorFunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)
W = V*Q

inflow = Expression(("uin*(1 - x[1]*x[1])", "0.0"), uin=U)
zero = Constant((0,)*ndim)

bcs = [
    DirichletBC(W.sub(0), zero, Top(1.0)),
    DirichletBC(W.sub(0), zero, Bottom(-1.0)),
    DirichletBC(W.sub(0), inflow, Left(0.0)),
    DirichletBC(W.sub(0), inflow, Right(6.0)),
]
if fix_pressure == 2:
    bcs.append(DirichletBC(W.sub(1), 0.0, Corner(6.0, -1.0),
                           method='pointwise'))

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

u0 = Function(V)
u0.vector().zero()

a = inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u0, v)*dx - p*div(v)*dx + \
    q*div(u)*dx
L = dot(zero, v)*dx

# # SUPG
# h = CellSize(mesh)
# tau = 1./sqrt(4*dot(u0, u0)/h**2 + 9*(4*mu/rho/h**2)**2)
# # fully explicit
# a_supg = tau*dot(grad(v)*u0, rho*grad(u)*u0 - div(mu*grad(u)) + grad(p))*dx

A = assemble(a)
b = assemble(L)
[bc.apply(A, b) for bc in bcs]

w = Function(W)
solve(A, w.vector(), b, 'mumps')
(u1, p1) = w.split(deepcopy=True)
print('errornorm: {0}'.format(errornorm(inflow, u1)))

# # adapt pressure constant  (compare against pressure boundary condition)
if fix_pressure == 1:
    p0 = p1.vector().array().mean()  # [0]
    p1.vector()[:] -= p0
    assign(w.sub(1), p1)

# different residuals
R = b - A*w.vector()
# Rm = rho*grad(u1)*u1 - div(mu*grad(u1)) + grad(p1)
# Rc = div(u1)
# print(assemble(sqrt(dot(Rm, Rm))*dx))
# print(assemble(sqrt(dot(Rc, Rc))*dx))
print('residual: {0}'.format(norm(R, 'l2')))

# plot(p1)

# w.vector()[:] *= 2

''' notes:
    pressure fixing:
    0: ignore pressure -> stokes resid ~ 1e-14, 10 it -> 1e-13, e ~ 1e-11/12
    1: substracting p[dof1] -> r~ 1e-12, 10 it -> 1e-12, e ~ 1e-12/13
    2: fix pressure by dBC on point -> ~ 1e-14, 10 it -> 1e-13/14, e ~ 1e-13
'''

# 1     Fixed point iteration: Picard x_k+1
# 2     Fixed point iteration: Picard update  (residual 1 ord smaller than 2)
# 3     Picard + Aitken
maxit = 10
# meth = 1
if meth == 1:
    it = 0
    # nc = rho*dot(grad(u0)*u, v)*dx
    while it < maxit:
        assign(u0, w.sub(0))
        A1 = assemble(a)
        [bc.apply(A1, b) for bc in bcs]
        solve(A1, w.vector(), b, 'mumps')

        if fix_pressure == 1:
            (_, p1) = w.split(deepcopy=True)
            p0 = p1.vector().array().mean()
            p1.vector()[:] -= p0
            assign(w.sub(1), p1)

        # true real one and only residual: .. grad(u)*u.
        assign(u0, w.sub(0))
        A1 = assemble(a)
        [bc.apply(A1, b) for bc in bcs]
        resid = b - A1*w.vector()
        print('{1} \t residual: {0}'.format(norm(resid, 'l2'), it))

        it += 1
    print('errornorm: {0}'.format(errornorm(inflow, w.sub(0))))

elif meth == 2:
    it = 0
    dw = Function(W)
    bcs = [
        DirichletBC(W.sub(0), zero, Top(1.0)),
        DirichletBC(W.sub(0), zero, Bottom(-1.0)),
        DirichletBC(W.sub(0), zero, Left(0.0)),
        DirichletBC(W.sub(0), zero, Right(6.0)),
    ]
    if fix_pressure == 2:
        bcs.append(DirichletBC(W.sub(1), 0.0, Corner(6.0, -1.0),
                               method='pointwise'))
    # nc = rho*dot(grad(u0)*u, v)*dx
    while it < maxit:
        assign(u0, w.sub(0))
        A1 = assemble(a)
        rhs = b - A1*w.vector()
        [bc.apply(A1, rhs) for bc in bcs]
        solve(A1, dw.vector(), rhs, 'mumps')

        if fix_pressure == 1:
            (_, dp1) = dw.split(deepcopy=True)
            dp0 = dp1.vector().array().mean()   # [0]
            dp1.vector()[:] -= dp0
            assign(dw.sub(1), dp1)

        w.vector().axpy(1.0, dw.vector())

        # true real one and only residual: .. grad(u)*u.
        assign(u0, w.sub(0))
        A1 = assemble(a)
        [bc.apply(A1, rhs) for bc in bcs]
        resid = b - A1*w.vector()
        print('{1} \t residual: {0}'.format(norm(resid, 'l2'), it))

        it += 1
    print('errornorm: {0}'.format(errornorm(inflow, w.sub(0))))

elif meth == 3:
    pass

(u, p) = w.split()
print('Re = {0}'.format(1.*U*2./0.035))
