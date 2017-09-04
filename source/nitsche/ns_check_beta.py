from dolfin import *
from solvers.ns_steady.nssolver import NSSolver, ReynoldsContinuation
from solvers.ns_steady.paramstudy import ParameterStudy
from solvers.ns_steady.nsproblem import NSProblem
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

inputfile = 'input/coarc.yaml'

pb = NSProblem(inputfile)

assert pb.options['boundary_conditions'][2]['method'] == 'nitsche'

# No slip essential
pb.options['boundary_conditions'][2]['method'] = 'essential'
pb.init()
sol = NSSolver(pb)
sol.solve()

u0, p0 = sol.w.split(deepcopy=True)

# No slip Nitsche
beta = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
param_modlist = [{'nitsche': {'beta1': x}} for x in beta]

ps = ParameterStudy(inputfile, param_modlist)
ps.solve()
# solutions stored in ps.sol

err_u = []
err_p = []
for sol in ps.sol:
    u, p = sol.split(deepcopy=True)
    err_u.append(norm(u.vector() - u0.vector())/norm(u0.vector()))
    err_p.append(norm(p.vector() - p0.vector())/norm(p0.vector()))

Re = 1500
plt.ion()
plt.figure()
plt.loglog(beta, err_u, '-o', beta, err_p, '-v')
plt.xlabel(r'$\beta_1$')
plt.ylabel('error')
plt.legend((r'$u$', r'$p$'))
plt.title(r'$Re = {0}$'.format(Re))

# tikz_save('results/ns_check_beta/err_Re{0}.tikz'.format(Re),
#           figurewidth='\\figurewidth', figureheight='\\figureheight')
