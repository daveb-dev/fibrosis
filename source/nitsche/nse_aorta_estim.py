''' Gamma slip BC for Navier-Stokes equations
    author: David Nolte
    email: dnolte@dim.uchile.cl
'''
# TODO: PCFieldSplitSetBlockSize -> sub blocks? ->
# https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex43.c.html,
# l. 1430

from dolfin import *
# petsc4py for petsc fieldsplit block preconditioner
# from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
# import scipy.linalg as la
# from sympy.utilities.codegen import ccode
# from functions.geom import *
from mpi4py import MPI

import os
from copy import deepcopy

from functions.utils import on_cluster
from solvers.ns_iteration import *

set_log_level(ERROR)
# parameters["allow_extrapolation"] = True
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["representation"] = "quadrature"
if on_cluster():
    parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -xHost -ipo"
else:
    # parameters['num_threads'] = 2
    parameters["form_compiler"]["cpp_optimize_flags"] = \
        "-O3 -ffast-math -march=native"


def getForms(W, bnds, opt):

    ndim = W.mesh().topology().dim()

    n = FacetNormal(W.mesh())
    h = CellSize(W.mesh())

    k = Constant(opt['mu'])
    rho = Constant(opt['rho'])
    beta = Constant(opt['beta'])
    beta2 = Constant(opt['beta2'])
    gamma = opt['gamma']
    if type(gamma) in [float, int]:
        gamma = Constant(gamma)

    elements = opt['elements']
    bctype = opt['bctype']
    wall = opt['wall']
    uwall = opt['uwall']
    symmetric = opt['symmetric']
    Rin = opt['Rin']
    pin = opt['pin']
    block_assemble = opt['block_assemble']
    uin = opt['uin']
    inlet = opt['inlet']
    sconv = Constant(opt['convection'])
    stem = Constant(opt['temam'])
    snewt = Constant(opt['nonlinear'] == 'newton')

    # boundary numbering
    # 1   inlet
    bnd_in = 1
    # 2   outlet
    bnd_out = 2
    # 3   wall
    bnd_w = 3
    # 4   slip/symmetry
    bnd_s = 4

    zero = Constant((0,)*ndim)

    bcs = []
    # inflow velocity profile
    if inlet == 'inflow':
        # y = a*(((xmax+xmin)/2 - x)**2 -(xmax-xmin)**2/4)
        if Rin:
            bmin = -Rin
            bmax = Rin
        else:
            # min/max coords on inlet boundary
            xmin = coords[:, 0].min()
            bmin = coords[coords[:, 0] == xmin, 1].min()
            bmax = coords[coords[:, 0] == xmin, 1].max()
        ys = (bmax + bmin) / 2.
        dy = (bmax - bmin) / 2.
        G = -uin/dy**2
        if ndim == 3:
            inflow = Expression(('G*(pow(ys - x[1], 2) + pow(ys - x[2], 2) - \
                                 pow(dy, 2))', '0.0', '0.0'),
                                ys=ys, dy=dy, G=G)
        elif ndim == 2:
            inflow = Expression(('G*(pow(ys - x[1], 2) + pow(ys - x[2], 2) - \
                                 pow(dy, 2))', '0.0'), ys=ys, dy=dy, G=G)
    elif inlet == 'pressure':
        if not type(pin) == dolfin.functions.constant.Constant:
            pin = Constant(pin)

    if inlet == 'inflow' and elements == 'Mini':
        parameters['krylov_solver']['monitor_convergence'] = False
        inflow = project(inflow, W.sub(0).collapse())

    # outflow: du/dn + pn = 0 on ds(2)

    # Define variational problem
    ds = Measure('ds', domain=W.mesh(), subdomain_data=bnds)

    F = zero

    def a(u, v):
        return inner(k*grad(u), grad(v))*dx

    def b(v, q):
        return div(v)*q*dx

    def f(v):
        return dot(F, v)*dx

    def t(u, p):
        return dot(k*grad(u), n) - p*n

    # TODO: only reasons to have the block wise assembly separate instead of a1
    # = a00 + a10 + a01, etc, are: ease of reading and maybe deprecate blocks
    # later
    if not block_assemble:
        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)

        u0 = Function(W.sub(0).collapse(), name='u0')

        # Standard form Stokes equations, Dirichlet RBs
        # choose b(u, q) positive so that a(u,u) > 0
        a1 = a(u, v) - b(v, p) + b(u, q)
        aconv = sconv*rho*dot(grad(u)*u0, v)*dx
        aconv += stem*sconv*0.5*rho*div(u0)*dot(u, v)*dx
        aconv_newton = snewt*sconv*rho*dot(grad(u0)*u, v)*dx
        aconv_newton += snewt*stem*sconv*0.5*rho*div(u)*dot(u0, v)*dx
        L1 = f(v)
        Lconv = dot(zero, v)*dx
        Lconv_newton = dot(zero, v)*dx

        # outflow: backflow stabilization
        if opt['backflow']:
            def abs_n(x):
                return 0.5*(x - abs(x))

            def abs_p(x):
                return 0.5*(x + abs(x))
            # bfstab = -sconv*rho*0.5*abs_n(dot(u0, n))*dot(u, v)
            # bfstab_newt = snewt*sconv*rho*0.5*abs_n(dot(u, n))*dot(u0, v)
            # FIXME: backflow term always, not only when u.n negative.
            # bfstab = -sconv*rho*0.5*dot(-u0, n)*dot(u, v)
            # bfstab_newt = -sconv*snewt*rho*0.5*dot(-u, n)*dot(u0, v)

            # XXX active backflow stab in true residual
            bfstab = -sconv*rho*0.5*abs_n(dot(u0, n))*dot(u, v)
            # FIXME "explicit" backflow stabilization in jacobian of residual
            bfstab_newt = Constant(0.)*dot(u, v)

            aconv += bfstab*ds(bnd_out)
            aconv_newton += bfstab_newt*ds(bnd_out)

        # inflow
        if inlet == 'inflow':
            if bctype == 'dirichlet':
                bcin = DirichletBC(W.sub(0), inflow, bnds, bnd_in)
                bcs.append(bcin)
            else:
                # stress tensor boundary integral
                a1 += - dot(k*grad(u)*n, v)*ds(bnd_in) \
                    + dot(p*n, v)*ds(bnd_in)
                # Nitsche BC
                a1 += beta/h*k*dot(u, v)*ds(bnd_in) \
                    + beta2/h*dot(u, n)*dot(v, n)*ds(bnd_in)
                L1 += beta/h*k*dot(inflow, v)*ds(bnd_in) \
                    + beta2/h*dot(inflow, n)*dot(v, n)*ds(bnd_in)
                # positive 'balance' terms
                a1 += + dot(u, k*grad(v)*n)*ds(bnd_in) \
                    - dot(u, q*n)*ds(bnd_in)
                L1 += + dot(inflow, k*grad(v)*n)*ds(bnd_in) \
                    - dot(inflow, q*n)*ds(bnd_in)
                if opt['backflow']:
                    aconv += -sconv*rho*0.5*abs_p(dot(u0, n)) * \
                        dot(u, v)*ds(bnd_in)
                    Lconv += -sconv*rho*0.5*abs_p(dot(u0, n)) * \
                        dot(inflow, v)*ds(bnd_in)
                    # FIXME: backflow term always, not only when u.n positive.
                    # aconv += -sconv*rho*0.5*dot(-u0, n)*dot(u, v)*ds(bnd_in)
                    # Lconv += -sconv*rho*0.5*dot(-u0, n) * \
                    #     dot(inflow, v)*ds(bnd_in)
                    # aconv_newton += -snewt*sconv*rho*0.5*dot(-u, n) * \
                    #     dot(u0, v)*ds(bnd_in)
                    # Lconv_newton += -snewt*sconv*rho*0.5*dot(-inflow, n) * \
                    #     dot(u0, v)*ds(bnd_in)
                    # FIXME: check those signs!
        elif inlet == 'pressure' and pin:
            # Pressure via Neumann BCs
            L1 += - dot(pin*n, v)*ds(bnd_in)
            if opt['backflow']:
                # aconv += -sconv*rho*0.5*abs_p(dot(u0, n)) * \
                #     dot(u, v)*ds(bnd_in)
                # Lconv += -sconv*rho*0.5*abs_p(dot(u0, n)) * \
                #     dot(inflow, v)*ds(bnd_in)
                # FIXME: backflow term always, not only when u.n positive.
                aconv += -sconv*rho*0.5*dot(u0, n)*dot(u, v)*ds(bnd_in)
                Lconv += -sconv*rho*0.5*abs_p(dot(u0, n)) * \
                    dot(inflow, v)*ds(bnd_in)
                aconv_newton += -snewt*sconv*rho*0.5*dot(-u, n) * \
                    dot(u0, v)*ds(bnd_in)
                Lconv_newton += -snewt*sconv*rho*0.5*dot(-inflow, n) * \
                    dot(u0, v)*ds(bnd_in)
                # FIXME: check those signs!

        if symmetric:     # assume symmetric boundary for bnd_s=4 !
            # symmetric BCs (==slip)
            a1 += - dot(k*grad(u)*n, n)*dot(v, n)*ds(bnd_s) \
                + p*dot(v, n)*ds(bnd_s)
            # Nitsche: u.n = 0
            if bctype == 'nitsche':
                a1 += beta/h*k*dot(u, n)*dot(v, n)*ds(bnd_s)
                # FIXME: add beta2/h*dot(u, n)*dot(v, n) here !??
                a1 += beta2/h*dot(u, n)*dot(v, n)*ds(bnd_s)
                # RHS -> zero
                # balance terms:
                a1 += + dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_s) \
                    - dot(u, n)*q*ds(bnd_s)
                if opt['backflow']:
                    aconv += bfstab*ds(bnd_s)
                    aconv_newton += bfstab_newt*ds(bnd_s)
            else:
                c0 = project(Constant(0), W.sub(0).sub(1).collapse())
                bcsym = DirichletBC(W.sub(0).sub(1), c0, bnds, bnd_s)
                bcs.append(bcsym)

        if wall == 'navierslip':
            # normal component of natural boundary integral
            a1 += - dot(k*grad(u)*n, n)*dot(v, n)*ds(bnd_w) \
                + p*dot(v, n)*ds(bnd_w)
            # tangential component (natural BC t.s.n = gamma*u.t)
            # a1 += - gamma*dot(u - dot(u, n)*n, v - dot(v, n)*n)*ds(bnd_w)
            a1 += - gamma*(dot(u, v) - dot(u, n)*dot(v, n))*ds(bnd_w)
            # Nitsche: u.n = 0
            a1 += beta/h*k*dot(u, n)*dot(v, n)*ds(bnd_w)
            # FIXME: add beta2/h*dot(u, n)*dot(v, b) here !??
            a1 += beta2/h*dot(u, n)*dot(v, n)*ds(bnd_w)
            # RHS -> zero
            # balance terms:
            a1 += + dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_w) \
                - dot(u, n)*q*ds(bnd_w)

            if opt['backflow']:
                aconv += bfstab*ds(bnd_w)
                aconv_newton += bfstab_newt*ds(bnd_w)

        if wall == 'none':
            # "do nothing" boundary condition
            a1 += - dot(k*grad(u)*n, v)*ds(bnd_w) \
                + dot(p*n, v)*ds(bnd_w)
            if opt['backflow']:
                aconv += bfstab*ds(bnd_w)
                aconv_newton += bfstab_newt*ds(bnd_w)

        if wall == 'noslip':
            if ndim == 3:
                noslip = Constant((uwall, 0.0, 0.0))
            elif ndim == 2:
                noslip = Constant((uwall, 0.0))

            if elements == 'Mini':
                parameters['krylov_solver']['monitor_convergence'] = False
                noslip = project(noslip, W.sub(0).collapse())

            if bctype == 'dirichlet':
                bc0 = DirichletBC(W.sub(0), noslip, bnds, bnd_w)
                bcs.append(bc0)
            elif bctype == 'nitsche':
                # stress tensor boundary integral
                a1 += - dot(k*grad(u)*n, v)*ds(bnd_w) \
                    + dot(p*n, v)*ds(bnd_w)
                # Nitsche: u = noslip
                a1 += beta/h*k*dot(u, v)*ds(bnd_w) \
                    + beta2/h*dot(u, n)*dot(v, n)*ds(bnd_w)
                L1 += beta/h*k*dot(noslip, v)*ds(bnd_w) \
                    + beta2/h*dot(noslip, n)*dot(v, n)*ds(bnd_w)
                # positive 'balance' stability terms
                a1 += + dot(u, k*grad(v)*n)*ds(bnd_w) \
                    - dot(u, q*n)*ds(bnd_w)
                L1 += dot(noslip, k*grad(v)*n)*ds(bnd_w) \
                    - dot(noslip, q*n)*ds(bnd_w)
                if opt['backflow']:
                    aconv += bfstab*ds(bnd_w)
                    aconv_newton += bfstab_newt*ds(bnd_w)

        tic()
        # A = assemble(a1)
        # b = assemble(L1)
        # [bc.apply(A, b) for bc in bcs]
        # print('assembly time:  %f' % toc())

    else:
        # ######### BLOCK ASSEMBLE ######### #
        # gives the same form, but split into blocks
        # NO mixed function space!
        # main purpose: inspect matrices with spy(A)
        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)

        u0 = Function(V, name='u0')

        a00 = a(u, v)
        aconv = sconv*rho*dot(grad(u)*u0, v)*dx + \
            0.5*sconv*rho*div(u0)*dot(u, v)*dx
        a10 = b(u, q)
        a01 = - b(v, p)
        # a11 = Constant(0)*p*q*dx

        L0 = f(v)
        L1 = Constant(0)*q*dx

        # XXX TODO: SIMPLIFY FORMS --> tt = (I-nn), etc
        # inflow
        if inlet == 'inflow':
            if bctype == 'dirichlet':
                bcin = DirichletBC(V, inflow, bnds, bnd_in)
                bcs.append(bcin)
            else:
                # stress tensor boundary integral
                a00 += - dot(k*grad(u)*n, v)*ds(bnd_in)
                a01 += dot(p*n, v)*ds(bnd_in)
                # Nitsche: u = inflow
                a00 += beta/h*k*dot(u, v)*ds(bnd_in)
                L0 += beta/h*k*dot(inflow, v)*ds(bnd_in)
                # 'balance' terms with oposite sign -> a(u, u) > 0 positive
                a00 += + dot(u, k*grad(v)*n)*ds(bnd_in)
                a10 += - dot(u, q*n)*ds(bnd_in)
                L0 += + dot(inflow, k*grad(v)*n)*ds(bnd_in)
                L1 += - dot(inflow, q*n)*ds(bnd_in)
        elif inlet == 'pressure' and pin:
            # Pressure via Neumann BCs
            # XXX ---> RHS!
            # a00 += - dot(k*grad(u)*n, v)*ds(bnd_in)
            # a01 += dot(p*n, v)*ds(bnd_in)
            L1 += - dot(pin*n, v)*ds(bnd_in)
            # Nitsche: u = inflow XXX: NOTHING
            # a00 += beta/h*k*dot(u, v)*ds(bnd_in)
            # L0 += beta/h*k*dot(inflow, v)*ds(bnd_in)
            # 'balance' terms with oposite sign -> a(u, u) > 0 positive
            # a00 += + dot(u, k*grad(v)*n)*ds(bnd_in)
            # a10 += - dot(u, q*n)*ds(bnd_in)
            # L0 += + dot(inflow, k*grad(v)*n)*ds(bnd_in)
            # L1 += - dot(inflow, q*n)*ds(bnd_in)

        if symmetric:     # assume symmetric boundary for bnd_s=4 !
            # symmetric BCs (==slip)
            # normal component of natural boundary integral
            a00 += - dot(k*grad(u)*n, n)*dot(v, n)*ds(bnd_s)
            a01 += p*dot(v, n)*ds(bnd_s)
            # Nitsche: u.n = 0
            a00 += beta/h*k*dot(u, n)*dot(v, n)*ds(bnd_s)
            # RHS -> zero
            # balance terms:
            a00 += + dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_s)
            a10 += - dot(u, n)*q*ds(bnd_s)

        if wall == 'navierslip':
            # normal component of natural boundary integral
            a00 += - dot(k*grad(u)*n, n)*dot(v, n)*ds(bnd_w)
            a01 += p*dot(v, n)*ds(bnd_w)
            # tangential component (natural BC t.s.n = gamma*u.t)
            a00 += - gamma*dot(u - dot(u, n)*n, v - dot(v, n)*n)*ds(bnd_w)
            # Nitsche: u.n = 0
            a00 += beta/h*k*dot(u, n)*dot(v, n)*ds(bnd_w)
            # RHS -> zero
            # balance terms:
            a00 += + dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_w)
            a10 += - dot(u, n)*q*ds(bnd_w)

        if wall == 'none':
            # "do nothing" boundary condition
            a00 += - dot(k*grad(u)*n, v)*ds(bnd_w)
            a01 += dot(p*n, v)*ds(bnd_w)

        if wall == 'noslip':
            if ndim == 3:
                noslip = Constant((uwall, 0.0, 0.0))
            elif ndim == 2:
                noslip = Constant((uwall, 0.0))

            if elements == 'Mini':
                parameters['krylov_solver']['monitor_convergence'] = False
                noslip = project(noslip, V)

            if bctype == 'dirichlet':
                bc0 = DirichletBC(V, noslip, bnds, bnd_w)
                bcs.append(bc0)
            elif bctype == 'nitsche':
                # stress tensor boundary integral
                a00 += - dot(k*grad(u)*n, v)*ds(bnd_w)
                a01 += dot(p*n, v)*ds(bnd_w)
                # Nitsche: u = noslip
                a00 += beta/h*k*dot(u, v)*ds(bnd_w)
                L0 += beta/h*k*dot(noslip, v)*ds(bnd_w)
                # skew symmetric 'balance' terms
                a00 += + dot(u, k*grad(v)*n)*ds(bnd_w)
                a10 += - dot(u, q*n)*ds(bnd_w)
                L0 += + dot(noslip, k*grad(v)*n)*ds(bnd_w)
                L1 += -dot(noslip, q*n)*ds(bnd_w)

        # A00 = assemble(a00)
        # B = assemble(a10)
        # BT = assemble(a01)
        # b0 = assemble(L0)
        # b1 = assemble(L1)

        # XXX does this work?
        # [bc.apply(A00, B, BT, b0, b1) for bc in bcs]

        # A = (A00, B, BT)
        # b = (b0, b1)
        # W = (V, Q)

    return a1, aconv, aconv_newton, L1, Lconv, Lconv_newton, bcs


def getForms_NonLin(W, bnds, opt):
    # beta=10, elements='TH', gamma=0., wall='noslip',
    # uwall=0.0, bctype='nitsche', symmetric=False,
    # block_assemble=False, inlet='inflow', Rref=None,
    # umax=1., pin=None, mu=0.01, rho=1.0):

    ndim = W.mesh().topology().dim()

    n = FacetNormal(W.mesh())
    h = CellSize(W.mesh())

    k = Constant(opt['mu'])
    rho = Constant(opt['rho'])
    beta = Constant(opt['beta'])
    beta2 = Constant(opt['beta2'])
    gamma = opt['gamma']
    if type(gamma) in [float, int]:
        gamma = Constant(gamma)

    elements = opt['elements']
    bctype = opt['bctype']
    wall = opt['wall']
    uwall = opt['uwall']
    symmetric = opt['symmetric']
    Rin = opt['Rin']
    pin = opt['pin']
    uin = opt['uin']
    inlet = opt['inlet']
    sconv = Constant(opt['convection'])

    # boundary numbering
    # 1   inlet
    bnd_in = 1
    # 2   outlet
    bnd_out = 2
    # 3   wall
    bnd_w = 3
    # 4   slip/symmetry
    bnd_s = 4

    zero = Constant((0,)*ndim)

    bcs = []
    # inflow velocity profile
    if inlet == 'inflow':
        # y = a*(((xmax+xmin)/2 - x)**2 -(xmax-xmin)**2/4)
        if Rin:
            bmin = -Rin
            bmax = Rin
        else:
            # min/max coords on inlet boundary
            xmin = coords[:, 0].min()
            bmin = coords[coords[:, 0] == xmin, 1].min()
            bmax = coords[coords[:, 0] == xmin, 1].max()
        ys = (bmax + bmin) / 2.
        dy = (bmax - bmin) / 2.
        G = -uin/dy**2
        if ndim == 3:
            inflow = Expression(('G*(pow(ys - x[1], 2) + pow(ys - x[2], 2) - \
                                 pow(dy, 2))', '0.0', '0.0'),
                                ys=ys, dy=dy, G=G)
        elif ndim == 2:
            inflow = Expression(('G*(pow(ys - x[1], 2) + pow(ys - x[2], 2) - \
                                 pow(dy, 2))', '0.0'), ys=ys, dy=dy, G=G)
    elif inlet == 'pressure':
        if not type(pin) == dolfin.functions.constant.Constant:
            pin = Constant(pin)

    if inlet == 'inflow' and elements == 'Mini':
        parameters['krylov_solver']['monitor_convergence'] = False
        inflow = project(inflow, W.sub(0).collapse())

    # outflow: du/dn + pn = 0 on ds(2)

    # Define variational problem
    ds = Measure('ds', domain=W.mesh(), subdomain_data=bnds)

    F = zero

    def a(u, v):
        return inner(k*grad(u), grad(v))*dx

    def b(v, q):
        return div(v)*q*dx

    def c(u, v):
        return dot(u, dot(k*grad(v), n))

    def f(v):
        return dot(F, v)*dx

    def t(u, p):
        return dot(k*grad(u), n) - p*n

    w = Function(W, name='w')
    # u, p = w.split(deepcopy=True)
    (u, p) = (as_vector((w[0], w[1])), w[2])  # 2D only
    # (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # u0 = Function(W.sub(0).collapse(), name='u0')

    # Standard form Stokes equations, Dirichlet RBs
    # choose b(u, q) positive so that a(u,u) > 0
    a1 = a(u, v) - b(v, p) + b(u, q)
    aconv = sconv*rho*dot(grad(u)*u, v)*dx
    aconv += sconv*0.5*rho*div(u)*dot(u, v)*dx
    L1 = f(v)
    Lconv = dot(zero, v)*dx

    # inflow
    if opt['backflow']:
        def abs_n(x):
            return 0.5*(x - abs(x))

        def abs_p(x):
            return 0.5*(x + abs(x))
        bfstab = -sconv*rho*0.5*abs_n(dot(u, n))*dot(u, v)
        aconv += bfstab*ds(bnd_out)
    if inlet == 'inflow':
        if bctype == 'dirichlet':
            bcin = DirichletBC(W.sub(0), inflow, bnds, bnd_in)
            bcs.append(bcin)
        else:
            # stress tensor boundary integral
            a1 += - dot(k*grad(u)*n, v)*ds(bnd_in) \
                + dot(p*n, v)*ds(bnd_in)
            # Nitsche BC
            a1 += beta/h*k*dot(u, v)*ds(bnd_in) \
                + beta2/h*dot(u, n)*dot(v, n)*ds(bnd_in)
            L1 += beta/h*k*dot(inflow, v)*ds(bnd_in) \
                + beta2/h*dot(inflow, n)*dot(v, n)*ds(bnd_in)
            # positive 'balance' terms
            a1 += + dot(u, k*grad(v)*n)*ds(bnd_in) \
                - dot(u, q*n)*ds(bnd_in)
            L1 += + dot(inflow, k*grad(v)*n)*ds(bnd_in) \
                - dot(inflow, q*n)*ds(bnd_in)
            if opt['backflow']:
                aconv += -sconv*rho*0.5*abs_p(dot(u, n)) * \
                    dot(u, v)*ds(bnd_in)
                Lconv += -sconv*rho*0.5*abs_p(dot(u, n)) * \
                    dot(inflow, v)*ds(bnd_in)
    elif inlet == 'pressure' and pin:
        # Pressure via Neumann BCs
        # XXX ---> RHS!
        # a00 += - dot(k*grad(u)*n, v)*ds(bnd_in)
        # a01 += dot(p*n, v)*ds(bnd_in)
        L1 += - dot(pin*n, v)*ds(bnd_in)
        # Nitsche: u = inflow XXX: NOTHING
        # a00 += beta/h*k*dot(u, v)*ds(bnd_in)
        # L0 += beta/h*k*dot(inflow, v)*ds(bnd_in)
        # 'balance' terms with oposite sign -> a(u, u) > 0 positive
        # a00 += + dot(u, k*grad(v)*n)*ds(bnd_in)
        # a10 += - dot(u, q*n)*ds(bnd_in)
        # L0 += + dot(inflow, k*grad(v)*n)*ds(bnd_in)
        # L1 += - dot(inflow, q*n)*ds(bnd_in)
        if opt['backflow']:
            aconv += -sconv*rho*0.5*abs_p(dot(u0, n)) * \
                dot(u, v)*ds(bnd_in)
            Lconv += -sconv*rho*0.5*abs_p(dot(u0, n)) * \
                dot(inflow, v)*ds(bnd_in)

    if symmetric:     # assume symmetric boundary for bnd_s=4 !
        # symmetric BCs (==slip)
        a1 += - dot(k*grad(u)*n, n)*dot(v, n)*ds(bnd_s) \
            + p*dot(v, n)*ds(bnd_s)
        # Nitsche: u.n = 0
        a1 += beta/h*k*dot(u, n)*dot(v, n)*ds(bnd_s)
        # FIXME: add beta2/h*dot(u, n)*dot(v, b) here !??
        a1 += beta2/h*dot(u, n)*dot(v, n)*ds(bnd_s)
        # RHS -> zero
        # balance terms:
        a1 += + dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_s) \
            - dot(u, n)*q*ds(bnd_s)
        if opt['backflow']:
            aconv += bfstab*ds(bnd_s)

    if wall == 'navierslip':
        # normal component of natural boundary integral
        a1 += - dot(k*grad(u)*n, n)*dot(v, n)*ds(bnd_w) \
            + p*dot(v, n)*ds(bnd_w)
        # tangential component (natural BC t.s.n = gamma*u.t)
        # a1 += - gamma*dot(u - dot(u, n)*n, v - dot(v, n)*n)*ds(bnd_w)
        a1 += - gamma*(dot(u, v) - dot(u, n)*dot(v, n))*ds(bnd_w)
        # Nitsche: u.n = 0
        a1 += beta/h*k*dot(u, n)*dot(v, n)*ds(bnd_w)
        # FIXME: add beta2/h*dot(u, n)*dot(v, b) here !??
        a1 += beta2/h*dot(u, n)*dot(v, n)*ds(bnd_w)
        # RHS -> zero
        # balance terms:
        a1 += + dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_w) \
            - dot(u, n)*q*ds(bnd_w)
        if opt['backflow']:
            aconv += bfstab*ds(bnd_w)

    if wall == 'none':
        # "do nothing" boundary condition
        a1 += - dot(k*grad(u)*n, v)*ds(bnd_w) \
            + dot(p*n, v)*ds(bnd_w)
        if opt['backflow']:
            aconv += bfstab*ds(bnd_w)

    if wall == 'noslip':
        if ndim == 3:
            noslip = Constant((uwall, 0.0, 0.0))
        elif ndim == 2:
            noslip = Constant((uwall, 0.0))

        if elements == 'Mini':
            parameters['krylov_solver']['monitor_convergence'] = False
            noslip = project(noslip, W.sub(0).collapse())

        if bctype == 'dirichlet':
            bc0 = DirichletBC(W.sub(0), noslip, bnds, bnd_w)
            bcs.append(bc0)
        elif bctype == 'nitsche':
            # stress tensor boundary integral
            a1 += - dot(k*grad(u)*n, v)*ds(bnd_w) \
                + dot(p*n, v)*ds(bnd_w)
            # Nitsche: u = noslip
            a1 += beta/h*k*dot(u, v)*ds(bnd_w) \
                + beta2/h*dot(u, n)*dot(v, n)*ds(bnd_w)
            L1 += beta/h*k*dot(noslip, v)*ds(bnd_w) \
                + beta2/h*dot(noslip, n)*dot(v, n)*ds(bnd_w)
            # positive 'balance' stability terms
            a1 += + dot(u, k*grad(v)*n)*ds(bnd_w) \
                - dot(u, q*n)*ds(bnd_w)
            L1 += dot(noslip, k*grad(v)*n)*ds(bnd_w) \
                - dot(noslip, q*n)*ds(bnd_w)
            if opt['backflow']:
                aconv += bfstab*ds(bnd_w)

    tic()
    # A = assemble(a1)
    # b = assemble(L1)
    # [bc.apply(A, b) for bc in bcs]
    # print('assembly time:  %f' % toc())

    return w, a1, aconv, L1, Lconv, bcs


def uwall(umax, R, y):
    return -umax/R**2*(y**2 - R**2)


def ref2measurement(uref, Wmeas=None, opt=None, dR=0.1):
    if not Wmeas and opt:
        # "stand alone" feature
        mesh_meas = prep_mesh(opt['mesh_meas'].format(dR, opt['hmeas']))
        Wmeas, bnds_meas = getFunctionSpaces(mesh_meas, 'P1')

    Warning('check if FS check is done right! -> ref2measurement()')
    Vm = Wmeas.sub(0).collapse()
    Vr = uref.function_space()
    if str(type(Vr.ufl_element())).split('.')[-1] == 'EnrichedElement':
        # interpolate to non-enriched FS on same mesh first, then project
        Vr = VectorFunctionSpace(Vr.mesh(), 'CG', 1)
        uref = interpolate(uref, Vr)
        umeas = project(uref, Vm)
    else:
        umeas = interpolate(uref, Vm)

    return umeas


def STEint(u0, W, mu, rho):
    ndim = W.mesh().topology().dim()
    mu = Constant(mu)
    rho = Constant(rho)

    (w, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    zero = Constant((0,)*ndim)
    noslip = project(zero, W.sub(0).collapse())
    bc = DirichletBC(W.sub(0), noslip, 'on_boundary')

    a = inner(grad(w), grad(v))*dx \
        - p*div(v)*dx + div(w)*q*dx

    L = - mu*inner(grad(u0), grad(v))*dx + rho*inner(grad(v)*u0, u0)*dx
    A, b = assemble_system(a, L, bc)  # , A_tensor=A_ste, b_tensor=b_ste)

    w1 = Function(W)
    solve(A, w1.vector(), b, 'mumps')

    _, p_est = w1.split(deepcopy=True)

    return p_est


def estimate_STEint(opt, params):
    opt_ref = opt.copy()
    opt_ref['wall'] = 'noslip'
    opt_ref['uwall'] = 0.0
    opt_ref['elements'] = 'Mini'

    dR_arr = params['dR_arr']

    uref, pref, dPref, resid = solve_aorta_ref(opt_ref)

    e_dP = []
    for i, dR in enumerate(dR_arr):
        mesh_meas = prep_mesh(opt['mesh_meas'].format(dR, opt['hmeas']))
        W, bnds = getFunctionSpaces(mesh_meas, 'P1')
        ds = Measure('ds', domain=W.mesh(), subdomain_data=bnds)

        umeas = ref2measurement(uref, Wmeas=W)

        plot(umeas)

        p_est = STEint(umeas, W, opt['mu'], opt['rho'])
        dP = assemble(p_est*ds(1) - p_est*ds(2))
        e_dP.append((dP - dPref)/dPref)

        print('\ndR/R: {0}'.format(dR/opt['Rin']))
        print('dP STEi:\t {0:g}'.format(dP))
        print('dP ref: \t {0:g}'.format(dPref))
        print('e(dP):  \t {0:g}'.format(e_dP[-1]))

        if on_cluster():
            # delete mesh from /dev/shm when done
            os.remove(mesh_meas)

    return e_dP


def solve_aorta_ref(opt):

    mesh_file_ref = opt['mesh_ref'].format(opt['href'])
    mesh_ref = prep_mesh(mesh_file_ref)

    tic()
    W, bnds = getFunctionSpaces(mesh_ref, opt['elements'])

    u1, p1, resid = solve_iteration(opt, W, bnds)

    # pressure jump
    ds = Measure('ds', domain=W.mesh(), subdomain_data=bnds)
    dPref = assemble(p1*ds(1) - p1*ds(2))

    # f1 = File("results/uref_estim.pvd")
    # f1 << uref
    # f1 = File("results/pref_estim.pvd")
    # f1 << pref

    unorm = norm(u1, "l2")
    pnorm = norm(p1, "l2")

    if MPI.COMM_WORLD.rank == 0:
        print('time elapsed:\t %fs' % toc())
        print('Pressure jump dP_ref: %f' % dPref)
        print("DOFS: %d" % W.dim())
        print("L2 norm velocity:\t %.6g" % unorm)
        print("L2 norm pressure:\t %.6g" % pnorm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    return u1, p1, dPref, resid


def getGammaSegments(opt, params, from_params=False):
    ''' Set Expression defining Gamma segmentation along boundary.
        If from_params is set to False, set all Gamma factors to 1.
        If True, chose first entry in corresponding parameter list.

        TODO: Write C++ JIT-compiled Expression for speed gain.
    '''
    ngs = params['num_gsegs']
    if from_params and type(params) == dict:
        Glst = [G[0] for G in params['gm_arr'][:ngs]]
    else:
        Glst = [1.0]*ngs

    dR = params['dR_arr'][0]
    xi = 1.0
    L0 = 2.0
    R = opt['Rin'] - dR
    gmbase = 2.0*opt['mu']*R/(R**2 - opt['Rin']**2)
    if ngs == 1:
        gamma_seg = Expression('G1*Gm', Gm=gmbase, G1=Glst[0])
    elif ngs == 2:
        gamma_seg = Expression(
            'x[0] < xi || x[0] > xi+L ? G1*Gm : G2*Gm',
            xi=xi, L=L0, Gm=gmbase, G1=Glst[0], G2=Glst[1])
    elif ngs == 3:
        gamma_seg = Expression(
            'x[0] < xi || x[0] > xi+L ? G1*Gm : ' +
            'x[0] < xi+L/2 ? G2*Gm : G3*Gm',
            xi=xi, L=L0, Gm=gmbase, G1=Glst[0], G2=Glst[1], G3=Glst[2])
    elif ngs == 4:
        gamma_seg = Expression(
            'x[0] < xi || x[0] > xi+L ? G1*Gm : ' +
            'x[0] < xi+L/3 ? G2*Gm : ' +
            'x[0] < xi+2*L/3 ?  G3*Gm : G4*Gm',
            xi=xi, L=L0, Gm=gmbase, G1=Glst[0], G2=Glst[1], G3=Glst[2],
            G4=Glst[3])
        # gamma_seg = Expression(
        #     'x[0] < xi || x[0] > xi+L ? G1*Gm : ' +
        #     'x[0] < xi+L/4 ? G2*Gm : ' +
        #     'x[0] < xi+3*L/4 ?  G3*Gm : G4*Gm',
        #     xi=xi, L=L0, Gm=gmbase, G1=Glst[0],
        #     G2=Glst[1], G3=Glst[2],
        #     G4=Glst[3])
    elif ngs == 5:
        gamma_seg = Expression(
            'x[0] < xi || x[0] > xi+L ? G1*Gm : ' +
            'x[0] < xi+L/4 ? G2*Gm : ' +
            'x[0] < xi+L/2 ?  G3*Gm : ' +
            'x[0] < xi+3*L/4 ?  G4*Gm : G5*Gm',
            xi=xi, L=L0, Gm=gmbase, G1=Glst[0], G2=Glst[1], G3=Glst[2],
            G4=Glst[3], G5=Glst[4])
    elif ngs == 6:
        gamma_seg = Expression(
            'x[0] < xi  ? G1*Gm : ' +
            'x[0] > xi+L ? G6*Gm : ' +
            'x[0] < xi+L/4 ? G2*Gm : ' +
            'x[0] < xi+L/2 ?  G3*Gm : ' +
            'x[0] < xi+3*L/4 ?  G4*Gm : G5*Gm',
            xi=xi, L=L0, Gm=gmbase, G1=Glst[0], G2=Glst[1], G3=Glst[2],
            G4=Glst[3], G5=Glst[4], G6=Glst[5])
    elif ngs == 0:
        gamma_seg = None
    else:
        gamma_seg = None
        error('Navier-Slip BC defined without corresponding Gamma \
              parameters! (ngs == {0})'.format(ngs))

    return gamma_seg


def estimate(opt, params):
    from itertools import product
    opt_ref = opt.copy()
    opt_ref['wall'] = 'noslip'
    opt_ref['uwall'] = 0.0
    opt_ref['elements'] = 'Mini'

    ngs = params['num_gsegs']
    dR_arr = params['dR_arr']
    # ui_arr = params['ui_arr']
    # Ri_arr = params['Ri_arr']

    # Base parameters
    gamma_seg = getGammaSegments(opt, params, from_params=False)
    tic()
    uref, pref, dPref, resid = solve_aorta_ref(opt_ref)
    print('solved ref in {0:.4g}s'.format(toc()))

    LI = LagrangeInterpolator()

    if opt['wall'] == 'navierslip':
        prm_grid = [g for g in params['gm_arr'][:ngs]]
        prm_grid.extend([params['ui_arr'], params['Ri_arr']])
    elif opt['wall'] == 'noslip':
        prm_grid = [params['ui_arr'], params['Ri_arr']]
    Np = np.prod([len(c) for c in prm_grid])
    Nall = Np*len(dR_arr)
    e_dP = []
    e_umeas = []
    i = 0

    for dR in dR_arr:
        mesh_meas = prep_mesh(opt['mesh_meas'].format(dR, opt['hmeas']))
        Wmeas, bnds_meas = getFunctionSpaces(mesh_meas, 'P1')
        ds_meas = Measure('ds', domain=Wmeas.mesh(), subdomain_data=bnds_meas)
        umeas = ref2measurement(uref, Wmeas=Wmeas)
        umeas_norm = norm(umeas.vector(), 'l2')
        (u1, p1) = Function(Wmeas).split(deepcopy=True)

        mesh_prm = prep_mesh(opt['mesh_prm'].format(dR, opt['h']))
        W, bnds = getFunctionSpaces(mesh_prm, opt['elements'])
        # TODO: assemble parts that are independent of parameters!
        # check speed gain
        for prms in product(*prm_grid):
            if ngs:
                gamma_seg.G1 = prms[0]
                if ngs >= 2:
                    gamma_seg.G2 = prms[1]
                if ngs >= 3:
                    gamma_seg.G3 = prms[2]
                if ngs >= 4:
                    gamma_seg.G4 = prms[3]
                if ngs >= 5:
                    gamma_seg.G5 = prms[4]
                opt['gamma'] = gamma_seg
            opt['uin'] = prms[-2]
            opt['Rin'] = prms[-1]

            tic()
            u, p, _ = solve_iteration(opt, W, bnds)

            ds = Measure('ds', domain=W.mesh(), subdomain_data=bnds)
            dPh = assemble(p*ds(1) - p*ds(2))
            LI.interpolate(u1, u)
            LI.interpolate(p1, p)

            e_umeas.append(norm(u1.vector() - umeas.vector(), 'l2')/umeas_norm)
            dP1 = assemble(p1*ds_meas(1) - p1*ds_meas(2))

            e_dP.append((dP1 - dPref)/dPref)

            i += 1
            if MPI.COMM_WORLD.rank == 0:
                # gcur = [{0:.2g}.format(gi) for gi in prms[0:-2]]
                gcur = ', '.join(['{0:.2g}'.format(gi) for gi in prms[0:-2]])
                print('run # {0}/{1} \t d = {2} \t g = ({3}), \
                      u = {4:.4g}, R = {5}'.format(i, Nall, dR, gcur,
                                                   prms[-2], prms[-1]))
                print('time elapsed:\t {0:.4g}s'.format(toc()))
                print('dP real:\t {0}'.format(dPh))
                print('dP meas:\t {0}'.format(dP1))
                print('e(dP) ref:\t {0} \t ***'.format(e_dP[-1]))
                print('e(u) meas: \t {0} \t ***'.format(e_umeas[-1]))

        if on_cluster():
            # delete mesh from /dev/shm when done
            os.remove(mesh_prm)
            os.remove(mesh_meas)

    err = (e_umeas, e_dP)
    save_npz(opt, params, err)

    return err, (u, p)


def solve_iteration(opt, W, bnds):
    ''' wrapper for different iterative solvers '''
    a, aconv, aconv_newton, L, Lconv, Lconv_newton, bcs = \
        getForms(W, bnds, opt)
    if opt['nonlinear'] == 'picard':
        u, p, resid = solve_picard(opt, a, aconv, L, Lconv, bcs, W)
    elif opt['nonlinear'] == 'aitken':
        u, p, resid = solve_aitken(opt, a, aconv, L, Lconv, bcs, W)
    elif opt['nonlinear'] == 'newton':
        u, p, resid = solve_newton(opt, a, aconv, aconv_newton, L, Lconv,
                                   Lconv_newton, bcs, W)
    elif opt['nonlinear'] == 'fenics':
        w, a, aconv, L, Lconv, bcs = getForms_NonLin(W, bnds, opt)
        u, p = solve_nonlinear(opt, w, a, aconv, L, Lconv, bcs, W, bnds)
        resid = None

    return u, p, resid


def solve_aorta(opt, params):

    opt['gamma'] = getGammaSegments(opt, params, from_params=True)
    # from_params - very dangerous option. NEEDS to be True for optimization

    mesh_prm = prep_mesh(opt['mesh_prm'].format(params['dR_arr'][0], opt['h']))
    W, bnds = getFunctionSpaces(mesh_prm, opt['elements'])

    u, p, resid = solve_iteration(opt, W, bnds)

    ds = Measure('ds', domain=W.mesh(), subdomain_data=bnds)
    dPh = assemble(p*ds(1) - p*ds(2))

    return u, p, dPh, resid


def optimize_aorta(x, umeas, opt, params):
    ''' callable solver function for scipy optimization methods
        x: vector of parameters (i.e. [g1,g2,g3,g4])

        procedure:
            minimize ||u-uref||/||uref||
            - compute uref
            - compute u for current set of parameters
            - compute and return error-norm
    '''

    print('xk = '+', '.join(['{0:.4g}'.format(xi) for xi in x]))
    if opt['wall'] == 'navierslip':
        g_arr = [[xi] for xi in x]
        params['gm_arr'] = g_arr
    else:
        opt['uin'] = x[0]
        opt['Rin'] = x[1]

    u, p, dPh, resid = solve_aorta(opt, params)

    # mesh_meas = prep_mesh(opt['mesh_meas'].format(dR, opt['hmeas']))
    # Wmeas, bnds_meas = getFunctionSpaces(mesh_meas, 'P1')
    Vmeas = umeas.function_space()
    # ds_meas = Measure('ds', domain=Wmeas.mesh(), subdomain_data=bnds_meas)
    LI = LagrangeInterpolator()
    u1 = Function(Vmeas)
    LI.interpolate(u1, u)
    # LI.interpolate(p1, p)

    # plot(u, title="u")
    # plot(umeas, title="umeas")

    # err = norm(u1.vector() - umeas.vector(), 'l2')**2 / \
    #     norm(umeas.vector(), 'l2')**2
    err = (norm(u1.vector() - umeas.vector(), 'l2')/norm(umeas.vector(),
                                                         'l2'))**2
    print('err(u) = {0},\t dPh = {1}'.format(err, dPh))

    return err


def scipy_optimize(opt, params):
    from scipy.optimize import minimize

    opt_ref = opt.copy()
    opt_ref['wall'] = 'noslip'
    opt_ref['uwall'] = 0.0
    opt_ref['elements'] = 'Mini'
    uref, pref, dPref, resid = solve_aorta_ref(opt_ref)

    params_opt = deepcopy(params)

    res = []
    for i, dR in enumerate(params['dR_arr']):
        params_opt['dR_arr'][0] = dR
        mesh_meas = prep_mesh(opt['mesh_meas'].format(dR, opt['hmeas']))
        Wmeas, bnds_meas = getFunctionSpaces(mesh_meas, 'P1')
# ds_meas = Measure('ds', domain=Wmeas.mesh(), subdomain_data=bnds_meas)
        umeas = ref2measurement(uref, Wmeas=Wmeas)

        if opt['wall'] == 'navierslip':
            x0 = np.ones(params['num_gsegs'])
            # x0 = np.array([0.6904, -4.449, 7.231, 0.8797])
            # causes instabilities
        else:
            x0 = np.ones(2)
        tmp = minimize(optimize_aorta, x0, method='BFGS', jac=False,
                       args=(umeas, opt, params_opt), options={'disp': True})
        res.append(tmp)

    print(res)

    return res


def save_vtk(opt, u, p):
    if opt['savevtk']:
        pass


def save_npz(opt, params, err):
    if opt['savenpz']:
        npzfile = opt['npzfile']
        npzpath = npzfile.split('/')[:-1]
        npzpath = '/'.join(npzpath)
        from functions.utils import trymkdir
        trymkdir(npzpath)
        np.savez(npzfile, params=params, err=err)


def load_npz(npzfile):
    npzpath = npzfile.split('/')[:-1]
    npzpath = '/'.join(npzpath)
    data = np.load(npzfile)
    params = data['params']
    err = data['err']
    return params, err


def cgs2mmHg(p):
    # 1 mmHg = 1*a Pa
    # 1 dyn/cm^2 = 0.1 Pa = 0.1/a mmHg
    a = 133.322387415
    p.vector()[:] *= 0.1/a
    return p


def plot_error_bars(prm_gm=None, prm_ns=None, err_gm=None, err_ns=None,
                    err_STEi=None):
    num = 3 - (err_gm, err_ns, err_STEi).count(None)
    barwidth = 1./(num + 1)
    atick = 0.5*num

    # plt.ion()
    plt.figure()
    ax1 = plt.subplot(111)
    plt.figure()
    ax2 = plt.subplot(111)
    k = 0
    if prm_gm and err_gm:
        dR_arr = prm_gm['dR_arr']
        ui_arr = prm_gm['ui_arr']
        Ri_arr = prm_gm['Ri_arr']
        g_arr = prm_gm['gm_arr']

        ngs = prm_gm['num_gsegs']
        prm_grid = [g for g in prm_gm['gm_arr'][:ngs]]
        prm_grid.extend([params['ui_arr'], params['Ri_arr']])

        shape = [len(dR_arr)]
        shape.extend([len(c) for c in prm_grid])
        e_umeas = err_gm[0]
        e_dP = err_gm[1]
        E_umeas = np.array(e_umeas).reshape(shape)
        E_dP = np.array(e_dP).reshape(shape)
        e_u_min = []
        e_dP_min = []
        index = np.arange(len(dR_arr))
        for i, dR in enumerate(dR_arr):
            e_u_min.append(E_umeas[i, :].min())
            imin = E_umeas[i, :].argmin()
            e_dP_min.append(E_dP[i, :].flat[imin])
            ij = np.unravel_index(imin, E_umeas[i, :].shape)
            gi_min = []
            gi_labels = ''
            for ig in range(ngs):
                gi_min = g_arr[ig][ij[ig]]
                gi_labels += (r'$\gamma_{0}^\star = {1}$'.
                              format(ig+1, gi_min) + '\n')
            # g_min.append(gi_min)
            # g1_min.append(g1_arr[ij[0]])
            # g2_min.append(g2_arr[ij[1]])

            ui_min = ui_arr[ij[ngs]]
            Ri_min = Ri_arr[ij[ngs+1]]

            uilabel = r'$u_i^\star = {0}$'.format(ui_min)
            Rilabel = r'$R_i^\star = {0}$'.format(Ri_min)

            x = index[i]
            y = e_u_min[-1]
            ax1.annotate(uilabel + '\n' + Rilabel + '\n' + gi_labels,
                         xy=(x, y), xytext=(10, 40),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='0.8',
                                   alpha=0.2))
            ax2.annotate(uilabel + '\n' + Rilabel + '\n' + gi_labels,
                         xy=(x, y), xytext=(10, 40),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='0.8',
                                   alpha=0.2))

        # plt.figure(1)
        ax1.bar(index, e_u_min, barwidth, label=r'$\gamma$-slip', color='b',
                alpha=0.4)
        # plt.figure(2)
        ax2.bar(index, e_dP_min, barwidth, label=r'$\gamma$-slip', color='b',
                alpha=0.4)
        k += 1

    if prm_ns and err_ns:
        dR_arr = prm_ns['dR_arr']
        ui_arr = prm_ns['ui_arr']
        Ri_arr = prm_ns['Ri_arr']

        shape = (len(dR_arr), len(ui_arr), len(Ri_arr))
        e_umeas = err_ns[0]
        e_dP = err_ns[1]
        E_umeas = np.array(e_umeas).reshape(shape)
        E_dP = np.array(e_dP).reshape(shape)
        e_u_min = []
        e_dP_min = []
        ui_min = []
        Ri_min = []
        index = np.arange(len(dR_arr))
        for i, dR in enumerate(dR_arr):
            e_u_min.append(E_umeas[i, :].min())
            imin = E_umeas[i, :].argmin()
            e_dP_min.append(E_dP[i, :].flat[imin])
            ij = np.unravel_index(imin, E_umeas[i, :].shape)
            ui_min.append(ui_arr[ij[0]])
            Ri_min.append(Ri_arr[ij[1]])

            uilabel = r'$u_i^\star = {0}$'.format(ui_min[-1])
            Rilabel = r'$R_i^\star = {0}$'.format(Ri_min[-1])

            # ax1.annotate(g1label, xy=(index[i], e_u_min[-1]),
            #              xytext=(-24, 36), textcoords='offset points')
            x = index[i] + k*barwidth
            y = e_dP_min[-1]
            ax1.annotate(uilabel + '\n' + Rilabel,
                         xy=(x, y), xytext=(0, 40),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='0.8',
                                   alpha=0.2))
            ax2.annotate(uilabel + '\n' + Rilabel,
                         xy=(x, y), xytext=(0, 40),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='0.8',
                                   alpha=0.2))

        # plt.figure(1)
        ax1.bar(index + k*barwidth, e_u_min, barwidth, label=r'noslip',
                color='r', alpha=0.4)
        # plt.figure(2)
        ax2.bar(index + k*barwidth, e_dP_min, barwidth, label=r'noslip',
                color='r', alpha=0.4)
        k += 1

    if err_STEi:
        dR_arr = prm_ns['dR_arr']
        index = np.arange(len(dR_arr))
        ax2.bar(index + k*barwidth, err_STEi, barwidth, label=r'STEint',
                color='g', alpha=0.4)

    ax1.set_xlabel(r'$\Delta R/R$')
    ax1.set_ylabel(r'$\mathcal{E}_{u}$')
    ax1.set_title(r'$L_2$ error of $\mathcal{I}_H^1(u)$ wrt measurement, ' +
                  r'$\tilde u\in \mathcal{P}_H^1$')
    ax1.set_xticks(index + atick*barwidth)
    ax1.set_xticklabels(dR_arr)
    ax1.set_ylabel(r'$\mathcal{E}_{u}$')
    ax1.legend(loc=2)

    ax2.set_xlabel(r'$\Delta R/R$')
    ax2.set_ylabel(r'$\mathcal{E}_{\Delta P}$')
    ax2.set_title(r'relative error of $\Delta P$ wrt reference solution')
    ax2.set_xticks(index + atick*barwidth)
    ax2.set_xticklabels(dR_arr)
    ax2.legend(loc=2)

    plt.show()


def plot_residuals(resid):
    ''' resid is expected to be list of residual vectors,
            resid = [np.array, np.array, ...]
    '''
    plt.ion()
    plt.figure(figsize=(6, 4))
    for res in resid:
        plt.semilogy(res)
    plt.xlabel('iterations')
    plt.ylabel('normalized residual')


if __name__ == '__main__':
    ''' different mesh types:
                coarc2d:    standard mesh, outlet length = 2x inlet
                coarc2ds:   shortened mesh, outlet=inlet
                coarc2dsf:  refinement along narrowing (h/2)
                coarc2dsf1: refinement around low point of narrowing (h/2)
            turns out coarc2ds with aitken has best/fastest convergence
    '''
    opt = {
        'mesh_ref': 'coarc2d_f0.6_d0_h{0:g}.h5',
        # 'mesh_ref': 'pipe2ds_d0_h{0:g}.h5',
        # 'mesh_ref': 'coarc3d_4_f0.6_ref_h{0:g}.h5',
        'mesh_prm': 'coarc2d_f0.6_d{0:g}_h{1:g}.h5',
        # 'mesh_prm': 'pipe2ds_d{0:g}_h{1:g}.h5',
        'mesh_meas': 'pipe2ds_d{0:g}_h{1:g}.h5',
        'href': 0.05,   # cm
        'hmeas': 0.05,
        'h': 0.05,
        'elements': 'Mini',
        'backflow': True,      # FIXME not implemented correctly for newton
        'convection': True,
        'temam': True,
        'bctype': 'nitsche',
        'beta': 1e6,               # needs to be large 1e6 ++ for coarc2ds_d0
        'beta2': 0,
        'wall': 'navierslip',
        'uwall': 0.0,
        'gamma': 0.0,
        'symmetric': True,
        'nonlinear': 'aitken',  # newton, picard, aitken, fenics, stokes
        'newton': {
            'use_corr': True,       # false -> can be faster, but more
            # unreliable due to cancellation(?)
            'use_aitken': True,
            'start_picard': 10
        },
        'report': 'progress',         # progress, summary, 'none'
        'maxit': 100,
        'tol': 1.0e-10,
        'inlet': 'inflow',
        'uin': 60.,     # in cm/s
        'Rin': 1.1,     # in cm
        'pin': 18,
        'block_assemble': False,
        'mu': 0.035,     # in P = g/cm/s
        'rho': 1.0,  # in g/cm^3  (1 g/cm^3 = 1000 kg/m^3)
        'plots': False,
        'savenpz': False,
        'npzfile': 'results/nse_estim/nse_gm_ui120.npz',
        'savevtk': False
    }
    opt_ns = opt.copy()
    opt_ns['wall'] = 'noslip'
    opt_ns['npzfile'] = 'results/nse_estim/nse_ns_ui120.npz'

    opt_ref = opt.copy()
    opt_ref['wall'] = 'noslip'
    opt_ref['uwall'] = 0.0
    opt_ref['bctype'] = 'nitsche'
    opt_ref['savenpz'] = False
    opt_ref['savevtk'] = True
    # opt_ref['inlet'] = 'pressure'
    # opt_ref['pin'] = 20000.

    params = {
        'dR_arr': np.array([0.0]),
        'num_gsegs': 1,
        'gm_arr': [
            [1.0]
            # np.arange(1.0, 20.0, 5),   # Pipe IN/OUT
            # np.arange(0.2, 20.0, 5),   # narrowing full/half/third/quarter
            # np.arange(0.2, 20.0, 5),   # narrowing 2nd half/third/quarter
            # np.arange(0.2, 20.0, 5),   # narrowing 3rd third/quarter
            # np.arange(0.2, 20.0, 5),   # narrowing 4th quarter
            # np.arange(0.2, 20.0, 5),  # PIPE OUT
        ],
        # 'ui_arr': opt['uin']*np.linspace(0.8, 1.2, 10),
        # 'Ri_arr': np.arange(0.95, 1.06, 0.025)
        'ui_arr': np.array([opt['uin']]),
        'Ri_arr': np.array([opt['Rin']])
    }
    params_ns = deepcopy(params)
    params_ns['num_gsegs'] = 0
    params_ns['gm_arr'] = []
    params_ns['ui_arr'] = []
    params_ns['Ri_arr'] = []

    u, p, dP, resid = solve_aorta(opt, params)
    # uref, pref, dPref, resid = solve_aorta_ref(opt_ref)

    ''' find optimal gamma(s) for Poiseuille flow '''
    '''
    res = scipy_optimize(opt, params)
    g_arr = [[xi] for xi in res[0]['x']]
    params['gm_arr'] = g_arr
    '''
    ''' check backflow influence '''
    '''
    opt['backflow'] = True
    u, p, dP, resid = solve_aorta(opt, params)
    opt['backflow'] = False
    u2, p2, dP2, resid = solve_aorta(opt, params)
    V = VectorFunctionSpace(u.function_space().mesh(), 'CG', 1)
    du = Function(V)
    ui = interpolate(u, V)
    u2i = interpolate(u2, V)
    du.assign(ui - u2i)
    print('u inf norm: {0}'.format(norm(du.vector(), 'linf')))
    p2 = interpolate(p2, p.function_space())
    dp = Function(p.function_space())
    dp.assign(p2 - p)
    print('p inf norm: {0}'.format(norm(dp.vector(), 'linf')))
    plot(du)
    plot(dp)
    '''
    ''' search optimal Nitsche betas:
        optimal values:
            beta1 = 1000 (upper limit! search more)
            beta2 = 0 or 10
            FIXME: OPTIMIZATION PROBLEM --> BFGS !?

            TODO: NEED TO TUNE BETA FOR EACH CASE... MESH/REYNOLDS wrt NOSLIP
            DIRICHLET           '''
    '''
    uref, pref, dPref, resid = solve_aorta_ref(opt_ref)
    V = VectorFunctionSpace(uref.function_space().mesh(), 'CG', 1)
    urefi = interpolate(uref, V)
    du = Function(V)
    dp = Function(pref.function_space())
    # beta1 = [10, 25, 50, 75, 100, 200, 500, 1000]
    beta1 = np.arange(10000000, 100000000, 10000000)
    # beta2 = [0, 10, 25, 50, 75, 100, 200, 500]
    beta2 = np.array([0])
    unorm, pnorm, dumax = [], [], []
    from itertools import product
    for b1, b2 in product(beta1, beta2):
        opt['beta'] = b1
        opt['beta2'] = b2
        uni, pni, dPni, resid = solve_aorta_ref(opt)
        unii = interpolate(uni, V)
        pnii = interpolate(pni, pref.function_space())
        dp.assign(pref - pnii)
        du.assign(urefi - unii)
        # plot(du)
        # plot(dp)
        dumax.append(norm(du.vector(), 'linf'))
        unorm.append(norm(du.vector(), 'l2')/norm(urefi.vector(), 'l2'))
        pnorm.append(norm(dp.vector(), 'l2')/norm(pref.vector(), 'l2'))
        print('\n{0} : b1 = {1}, b2 = {2}'.format(opt['bctype'], opt['beta'],
                                                  opt['beta2']))
        print('U error L2-norm:\t{0}'.format(unorm[-1]))
        print('U error INF:  \t\t{0}'.format(dumax[-1]))
        print('P error L2-norm:\t{0}\n'.format(pnorm[-1]))

    shape = (len(beta1), len(beta2))
    unorm = np.array(unorm)
    pnorm = np.array(pnorm)
    iminu = np.argmin(unorm)
    iju = np.unravel_index(iminu, shape)
    b1optu = beta1[iju[0]]
    b2optu = beta2[iju[1]]
    print('E(u) min = {0}  for  b1 = {1}, b2 = {2},  (p = {3}, dumax = {4})'.
          format(unorm[iminu], b1optu, b2optu, pnorm[iminu], dumax[iminu]))
    iminp = np.argmin(pnorm)
    ijp = np.unravel_index(iminp, shape)
    b1optp = beta1[ijp[0]]
    b2optp = beta2[ijp[1]]
    print('E(p) min = {0}  for  b1 = {1}, b2 = {2},  (u = {3}, dumax = {4})'.
          format(pnorm[iminp], b1optp, b2optp, unorm[iminp], dumax[iminp]))
    '''

    ''' Poiseuille pipe flow benchmark
        use mesh pipe2ds_*.h5 '''
    '''
    uref, pref, dPref, resid = solve_aorta_ref(opt_ref)
    # uref, pref, dPref, resid = solve_aorta(opt, params)
    # compare with analytic Poiseuille solution
    if opt['inlet'] == 'inflow':
        inflow = Expression(("-Ui*(x[1]*x[1]-R)", "0.0"), Ui=opt['uin'],
                            R=opt['Rin'])
        if opt['elements'] == 'Mini':
            V1 = VectorFunctionSpace(uref.function_space().mesh(), 'CG', 1)
            u1 = interpolate(uref, V1)
        else:
            V1 = uref.function_space()
            u1 = uref
        uex = interpolate(inflow, V1)
        du = Function(V1)
        du.assign(uex - u1)
        plot(du)
        unorm = norm(du.vector(), 'linf')/norm(uex.vector(), 'linf')
        print('\nU error inf-norm:  {0}'.format(unorm))
    else:
        P = Expression("Pi*(1 - x[0]/4.)", Pi=opt['pin'])
        pex = interpolate(P, pref.function_space())
        dp = Function(pref.function_space())
        dp.assign(pex - pref)
        plot(dp)
        pnorm = norm(dp.vector(), 'linf')/norm(pex.vector(), 'linf')
        print('\nP error inf-norm:  {0}'.format(pnorm))

    print('elements:  {0} \t nonlin:  {1}'.format(opt['elements'],
                                                  opt['nonlinear']))
    '''
    ''' Debugging notes.
        hi
    '''
    # res = scipy_optimize(opt, params)
    # g_arr = [[xi] for xi in res[0]['x']]
    # params['gm_arr'] = g_arr

    # FIXME check f1 mesh AND ERROR SQUARED

    # example WITH BACKFLOW
    # g_arr = [[0.99471772462129593], [0.99273456633090973],
    #          [0.80734307318925858], [0.77656416594982147],
    #          [0.99984055757522583]]
    # params['gm_arr'] = g_arr

    # g_arr = [[.76273233], [3.46096297], [-11.1773951], [4.81003005],
    #          [0.15572145]]
    # params['gm_arr'] = g_arr

    # uin = 60, 6 gammas, h=href=0.05, H=1.5
    # x: array([ 0.97714379,  0.91196416, -0.02543219,  0.54205487,
    #   0.99900856, 1.02545351])

    # uin = 60, 4 gammas, h=href=H=0.05
    # start: err^2 = 0.00993796372466
    # end: err^2 = 0.009426
    # x: array([ 1.59391363, -2.76638063,  0.60485336,  0.80202479])

    # uin = 60, 4 gammas, h=href=0.05, H=0.15
    # start: err^2 = 0.0251179499139
    # end: err^2 = 0.159 NO CONVERGENCE

    # uin = 60, 4 gammas, dR=0.15, h=0.05, H=0.15
    # x: array([ 0.91819149,  0.6774945 , -3.22926412,  1.01302564])
    # e(dP) ref:       -0.21715335208
    # e(u) meas:       0.213710921399

    # uin = 60, 4 gammas, dR=0.2, h=0.05, H=0.15
    # x: array([ 1.16715696,  0.98906183,  1.04220653,  1.03568944])
    # e(dP) ref:       0.436300378586
    # e(u) meas:       0.387749726879

    # XXX do: channel-stokes estimate, channel-NSE (same) ... debug

    # f1 = File('results/nse_estim/u_dum_d0.15.pvd')
    # # f2 = File('results/nse_estim/p_refs_ui120.pvd')
    # uref.rename("velocity", "u")
    # # pref.rename("pressure", "p")
    # f1 << uref
    # # f2 << pref

    # res2 = scipy_optimize(opt_ns, params_ns)[0]
    # uin = 60, h=href=0.05, H=1.5
    # x: array([ 36.54791804,  25.91316507])

    # res2 = {'x': np.array([36.54791804, 25.91316507])}
    # params_ns['ui_arr'] = [res2['x'][0]]
    # params_ns['Ri_arr'] = [res2['x'][1]]

    # opt_ref['nonlinear'] = 'picard'
    # uref_p, pref_p, dPref_p, resid = solve_aorta_ref(opt_ref)
    # opt_ref['nonlinear'] = 'aitken'
    # uref_a, pref_a, dPref_a, resid = solve_aorta_ref(opt_ref)
    # opt_ref['nonlinear'] = 'fenics'
    # uref_n, pref_n, dPref_n, resid = solve_aorta_ref(opt_ref)

    # eu1 = norm(uref_p.vector() - uref_a.vector(), 'l2')/norm(uref_p.vector(),
    #                                                          'l2')
    # eu2 = norm(uref_p.vector() - uref_n.vector(), 'l2')/norm(uref_p.vector(),
    #                                                          'l2')
    # eu3 = norm(uref_n.vector() - uref_a.vector(), 'l2')/norm(uref_a.vector(),
    #                                                          'l2')
    # ep1 = norm(pref_p.vector() - pref_a.vector(), 'l2')/norm(pref_p.vector(),
    #                                                          'l2')
    # ep2 = norm(pref_p.vector() - pref_n.vector(), 'l2')/norm(pref_p.vector(),
    #                                                          'l2')
    # ep3 = norm(pref_n.vector() - pref_a.vector(), 'l2')/norm(pref_a.vector(),
    #                                                          'l2')
    # print('Picard - Aitken:   e_u = {0} \t e_p = {1}'.format(eu1, ep1))
    # print('Picard - Newton:   e_u = {0} \t e_p = {1}'.format(eu2, ep2))
    # print('Newton - Aitken:   e_u = {0} \t e_p = {1}'.format(eu3, ep3))

    # # umeas = ref2measurement(uref, Wmeas=None, opt=opt, dR=0.1)

    # err_gm, (u, p) = estimate(opt, params)
    # plot_error_bars(prm_gm=params, err_gm=err_gm)
    # opt_ns['gamma'] = None
    # err_ns, (uns, _) = estimate(deepcopy(opt_ns), params_ns)

    # e_dP_stei = estimate_STEint(opt, params)

    # plot_error_bars(prm_gm=params, prm_ns=params_ns, err_gm=err_gm,
    #                 err_ns=err_ns, err_STEi=e_dP_stei)
    # plot_error_bars(prm_ns=params_ns, err_ns=err_ns)

    # plt.figure()
    # plt.plot(err_gm[0], err_gm[1], '-o')
    # plt.xlabel('err(u)')
    # plt.ylabel(r'err($\Delta p$)')
    # plt.title(r'err u/dP dependency for variation over $\gamma_1,\gamma_2$')
    # plt.show()
