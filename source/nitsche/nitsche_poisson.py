from dolfin import *
import numpy as np
import sympy as sp
import scipy.linalg as la
from sympy.utilities.codegen import ccode
import matplotlib.pyplot as plt
from functions.geom import *


def poisson2D(mesh, u=None, beta=10, plots=False, p=2, solver='minres',
              prec='amg'):
    if u:
        fsym = (-u.diff(x, 2) - u.diff(y, 2)).simplify()
        f = Expression(ccode(fsym).replace('M_PI', 'pi'))
        u_ex = Expression(ccode(u).replace('M_PI', 'pi'))
    elif not f:
        f = Constant(0)
    else:
        # f given as arg
        pass

    beta = Constant(beta)

    V = FunctionSpace(mesh, "CG", 1)
    timing = []
    # left = Left(-R)
    # right = Right(R)
    # top = Top(Ly)
    # bottom = Bottom()

    # bnds = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    # bnds.set_all(0)
    # left.mark(bnds, 0)
    # right.mark(bnds, 0)
    # top.mark(bnds, 0)
    # bottom.mark(bnds, 0)

    ui = interpolate(u_ex, V)
    bcs = DirichletBC(V, ui, 'on_boundary')

    u = TrialFunction(V)
    v = TestFunction(V)

    n = FacetNormal(mesh)
    h = CellSize(mesh)

    # strong DBC
    a1 = inner(grad(u), grad(v))*dx
    L1 = f*v*dx

    w1 = Function(V)
    tic()
    solve(a1 == L1, w1, bcs, solver_parameters={"linear_solver": solver,
                                                "preconditioner": prec})
    timing.append(toc())

    # Nitsche form, symmetric
    a2 = inner(grad(u), grad(v))*dx - dot(dot(grad(u), n), v)*ds - \
        dot(u, dot(grad(v), n))*ds + beta/h*u*v*ds   # Nitsche terms
    L2 = f*v*dx - dot(ui, dot(grad(v), n))*ds + beta/h*ui*v*ds

    w2 = Function(V)
    tic()
    solve(a2 == L2, w2, solver_parameters={"linear_solver": solver,
                                           "preconditioner": prec})
    timing.append(toc())

    # Nitsche form, non symmetric, pos def
    a3 = inner(grad(u), grad(v))*dx - dot(dot(grad(u), n), v)*ds + \
        dot(u, dot(grad(v), n))*ds + beta/h*u*v*ds   # Nitsche terms
    L3 = f*v*dx + dot(ui, dot(grad(v), n))*ds + beta/h*ui*v*ds

    w3 = Function(V)
    tic()
    solve(a3 == L3, w3, solver_parameters={"linear_solver": solver,
                                           "preconditioner": prec})
    timing.append(toc())

    # Nitsche without 'extra terms'
    a4 = inner(grad(u), grad(v))*dx - dot(dot(grad(u), n), v)*ds + \
        beta/h*u*v*ds   # Nitsche terms
    L4 = f*v*dx + beta/h*ui*v*ds

    w4 = Function(V)
    tic()
    solve(a4 == L4, w4, solver_parameters={"linear_solver": solver,
                                           "preconditioner": prec})
    timing.append(toc())

    if plots:
        plot(ui, title='exact solution')
        plot(w1, title='Dirichlet')
        plot(w2, title='Nitsche symmetric (standard)')
        plot(w3, title='Nitsche non symmetric')
        plot(w4, title='Nitsche without extra terms')
        interactive()

    err = [errornorm(ui, w1), errornorm(ui, w2), errornorm(ui, w3),
           errornorm(ui, w4)]

    z = ui.vector().array()
    d1 = w1.vector().array() - z
    d2 = w2.vector().array() - z
    d3 = w3.vector().array() - z
    d4 = w4.vector().array() - z
    u_norm = la.norm(z, p)
    err2 = [la.norm(d1, p)/u_norm, la.norm(d2, p)/u_norm,
            la.norm(d3, p)/u_norm, la.norm(d4, p)/u_norm]

    dofs = V.dim()
    return err, err2, dofs, timing


beta = 1000
ql = range(4, 9)
h = []
dofs = []
err = np.zeros((len(ql), 8))
pord = 2
for i, q in enumerate(ql):
    R = 1.0
    Ly = 1.0
    # mesh = tube_mesh2D(1, Ly, 2**q)
    mesh = UnitSquareMesh(2**q, 2**q)
    h.append(mesh.hmin())

    x, y = sp.symbols('x[0] x[1]')
    u_ex = sp.exp(x**2 + y**2)*(x**2 + y**2)
    # u_ex = x**2 + y**2
    # u_ex = sp.sin(x*sp.pi/R)*sp.sin(y*sp.pi/Ly)
    e1, e2, dof, tt = poisson2D(mesh,
                                u=u_ex,
                                beta=beta,
                                plots=False,
                                p=pord,
                                solver='tfqmr',  # gmres, lu,..
                                prec='amg')
    err[i, :] = np.hstack((e1, e2))
    dofs.append(dof)

print('beta = %d' % beta)
print('DOFs\t|\t\t energy norm ||u-wi||\t\t\t|\t\t %d-norm ||u-wi||' % pord)
for i, dof in enumerate(dofs):
    print('%i\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f' % (dof,
                                                                  err[i, 0],
                                                                  err[i, 1],
                                                                  err[i, 2],
                                                                  err[i, 3],
                                                                  err[i, 4],
                                                                  err[i, 5],
                                                                  err[i, 6],
                                                                  err[i, 7]))
print('\ntime:\t%f\t%f\t%f\t%f' % (tt[0], tt[1], tt[2], tt[3]))

plt.ion()
# plt.figure()
# plt.loglog(dofs, err[:,0:4], '-o')
# plt.title('energy norm')
# plt.xlabel('DOFs')
# plt.ylabel(r'$|||e|||$')
# plt.legend(('Dirichlet', 'Nitsche std', 'Nitsche SPD', 'Nitsche w/o terms'),
#            loc='lower left')
# plt.axis('scaled')

plt.figure()
plt.loglog(dofs, err[:, 4:8], '-o')
plt.title('Poisson problem, relative error, 2 norm')
plt.xlabel('DOFs')
plt.ylabel(r'$||e||_2/||u||_2$')
plt.legend(('Dirichlet', 'Nitsche standard', 'Nitsche PosDef',
            'Nitsche w/o extra'), loc='lower left')
plt.axis('scaled')
