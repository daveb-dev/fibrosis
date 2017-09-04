''' For the coarc case with Navier-Slip BCs, compare the integral of u.n
    over the boundary for different values of beta1. Check the influence of
    backflow stabilization on the Navier-Slip boundary.
'''
from dolfin import *
from solvers.ns_steady.nssolver import NSSolver, ReynoldsContinuation
from solvers.ns_steady.paramstudy import ParameterStudy
from solvers.ns_steady.nsproblem import NSProblem
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import itertools
import numpy as np

inputfile = 'input/coarc_ubnd.yaml'
pb = NSProblem(inputfile)
pb.init_mesh()
ds = Measure('ds', domain=pb.mesh, subdomain_data=pb.bnds)
LI = LagrangeInterpolator()
V = VectorFunctionSpace(pb.mesh, 'CG', 1)
ui = Function(V)
v = TestFunction(V)
n = FacetNormal(pb.mesh)

input_ref = 'input/coarc_ubnd_ref.yaml'
pb_ref = NSProblem(input_ref)
pb_ref.init()
sol_ref = NSSolver(pb_ref)
sol_ref.solve()
uref, pref = sol_ref.w.split(deepcopy=True)
urefi = Function(V)
LI.interpolate(urefi, uref)
un_ref = assemble(dot(urefi, n)*dot(v, n)*ds(4))
ut_ref = assemble((dot(urefi - dot(urefi, n)*n, v - dot(v, n)*n))*ds(4))

# Navier-Slip Nitsche WITHOUT backflow stab
beta = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
# beta = [1e3, 1e5, 1e6]
# bfstab = [0, 1, 2]
bfstab = [2]
param_modlist = [{'nitsche': {'beta1': x}, 'backflowstab': {'nitsche': y}} for
                 (y, x) in itertools.product(bfstab, beta)]

ps = ParameterStudy(inputfile, param_modlist)
ps.solve()
# solutions stored in ps.sol


# INTEGRATE u.n and u.t
uxn = np.zeros(len(param_modlist))
uxt = np.zeros(len(param_modlist))
un = []
ut = []
# ut = np.zeros(len(param_modlist))
# uxn2 = np.zeros(len(param_modlist))
for i, sol in enumerate(ps.sol):
    u, p = sol.split(deepcopy=True)
    LI.interpolate(ui, u)
    uxn[i] = assemble(abs(dot(ui, n))*ds(4))
    uxt[i] = assemble(sqrt(dot(ui - dot(ui, n)*n, ui - dot(ui, n)*n))*ds(4))
    # uxn2[i] = sqrt(assemble(dot(ui, n)**2*ds(4)))
    un.append(assemble(dot(ui, n)*dot(v, n)*ds(4)))
    ut.append(assemble((dot(ui - dot(ui, n)*n, v - dot(v, n)*n))*ds(4)))


def extract_bnd(uvec, bnds, bid, V):
    It_facet = SubsetIterator(bnds, bid)
    X = []
    vid = []
    for f in It_facet:
        for v in vertices(f):
            X.append([v.point().x(), v.point().y()])
            vid.append(v.index())
    vid, idx = np.unique(vid, return_index=True)
    # FIXME: need to account for sub spaces.. !?!?
    # https://fenicsproject.org/qa/6943/vertex_to_dof_map-and-dof_to_vertex_map-work-mixedspace
    n = V.dofmap().num_entity_dofs(0)   # dimension
    v2d = vertex_to_dof_map(V)
    dofs_u = v2d[vid*2]
    dofs_v = v2d[vid*2 + 1]
    ubnd = uvec[dofs_u]
    vbnd = uvec[dofs_v]
    X = np.array(X)[idx, :]
    return X, ubnd, vbnd, dofs_u, dofs_v


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a, idx = np.unique(a.view([('', a.dtype)]*a.shape[1]),
                              return_index=True)
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1])), idx

col = itertools.cycle(['k', 'r', 'b', 'g', 'm', 'y', 'c'])
plt.ion()
fignum = plt.get_fignums()
fignum = max(fignum) if fignum else 0

ut.append(ut_ref)
un.append(un_ref)

for i, (ut_, un_) in enumerate(zip(ut, un)):
    X, utbnd, vtbnd, dofs_u, dofs_v = extract_bnd(ut_, pb.bnds, 4, V)
    _, unbnd, vnbnd, _, _ = extract_bnd(un_, pb.bnds, 4, V)
    col_ = col.next()
    plt.figure(fignum+1)
    plt.title('tangential velocity')
    plt.quiver(X[:, 0], X[:, 1] + 0.2*i, utbnd, vtbnd, scale=2, alpha=0.5,
               color=col_)
    plt.figure(fignum+2)
    plt.title('magnitude of normal velocity')
    plt.scatter(X[:, 0], np.sqrt(unbnd**2 + vnbnd**2),
                color=col_)
    plt.figure(fignum+3)
    plt.title('magnitude of tangential velocity')
    plt.scatter(X[:, 0], np.sign(utbnd)*np.sqrt(utbnd**2 + vtbnd**2),
                color=col_)
    plt.figure(fignum+4)
    plt.title('magnitude of wall velocity')
    plt.scatter(X[:, 0], np.sqrt((unbnd + utbnd)**2 + (vnbnd + vtbnd)**2),
                color=col_)
plt.figure(fignum+2).gca().grid()
plt.figure(fignum+3).gca().grid()
plt.figure(fignum+4).gca().grid()
legstr = [r'$\beta = {0:1.1e}$'.format(x) for x in beta]
legstr.append('reference')
for f in range(fignum+2, fignum+5):
    plt.figure(f).gca().legend(legstr)
    # plt.figure(f)
    # plt.legend(legstr)

# integral over boundary
# uxn = np.reshape(uxn, (len(bfstab), len(beta)))
# uxt = np.reshape(uxt, (len(bfstab), len(beta)))
# uxn2 = np.reshape(uxn2, (len(bfstab), len(beta)))


#
'''
Re = 1000
plt.ion()
plt.figure()
plt.semilogx(beta, uxn.T, 'o')
plt.xlim((min(beta)/2., max(beta)*2.))
# plt.semilogx(beta, uxn2.T, '-s', lw=2, ms=4)
plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\langle u\cdot n\rangle$')
plt.legend(('no bfstab', r'$u\cdot n=0$', r'$|u\cdot n| = 0$'), loc=0)
# plt.legend(('abs', 'rms'), loc=0)
plt.title(r'$Re = {0}$, normal flow'.format(Re))

# tikz_save('results/ns_check_beta/err_Re{0}.tikz'.format(Re),
#           figurewidth='\\figurewidth', figureheight='\\figureheight')

plt.figure()
plt.semilogx(beta, uxt.T, 'o')
plt.xlim((min(beta)/2., max(beta)*2.))
plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\langle u\cdot t\rangle$')
plt.legend(('no bfstab', r'$u\cdot n=0$', r'$|u\cdot n| = 0$'), loc=0)
# plt.legend(('abs', 'rms'), loc=0)
plt.title(r'$Re = {0}$, tangential flow'.format(Re))

plt.figure()
plt.semilogx(beta, (uxn/uxt).T, 'o')
plt.xlim((min(beta)/2., max(beta)*2.))
plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\langle u\cdot n\rangle/\langle u\cdot t\rangle$')
plt.legend(('no bfstab', r'$u\cdot n=0$', r'$|u\cdot n| = 0$'), loc=0)
# plt.legend(('abs', 'rms'), loc=0)
plt.title(r'$Re = {0}$, norm/tang'.format(Re))
'''
