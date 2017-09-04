''' Steady Navier-Stokes solver module

Author: David Nolte (dnolte@dim.uchile.cl)
Date:   2016-09-22
'''
from dolfin import *
import numpy as np
import warnings


class GeneralProblem(NonlinearProblem):
    ''' This class represents a nonlinear variational problem:
        Find w in W s.t.
            F(w; z) = 0  f.a. z in W^

        GeneralProblem is a subclass of dolfin.NonlinearProblem.

    '''
    def __init__(self, F, w, J=None, bcs=[]):
        ''' Create a nonlinear variational problem for the given form.

        If the Jacobian is not given, it is computed automatically.

        Args:
            F               variational form of residual
            w               solution function
            J (optional)    variational form of Jacobian or its approximation
            bcs (optional)  list of essential boundary conditions
        '''
        NonlinearProblem.__init__(self)
        self.fform = F
        self.w = w
        self.bcs = bcs
        if J:
            self.jacobian = J
        else:
            self.jacobian = derivative(F, w)

        pass

    def F(self, b, x):
        ''' Assemble residual vector and apply boundary conditions, if given.
        The boundary conditions for the increment are set as the difference
        between the problem's boundary conditions and the current iterate.

        Args:
            b       target PETScVector
            x       current solution PETScVector
        '''
        assemble(self.fform, tensor=b)
        [bc.apply(b, x) for bc in self.bcs]
        pass

    def J(self, A, x):
        ''' Assemble the Jacobian matrix and set boundary conditions, if given.

        Args:
            A       target PETScMatrix
            x       current solution PETScVector
        '''
        assemble(self.jacobian, tensor=A)
        [bc.apply(A) for bc in self.bcs]
        pass


class AitkenAccelerator:
    ''' Class for Aitken accelerator, based on velocity increment. Stores
    one previous velocity increment and relaxation parameter and calculates
    new omega.  Used in order to minimize code clustering.
    '''
    def __init__(self, options):
        ''' Initialize Aitken accelerator.

        Args:
            options     options dictionary

        Attributes:
            use (int):  switch, 1: enable Aitken for Picard iteration
                                2: enable Aitken at all times
                                0: don't use Aitken acceleration
            picard:     True if picard iteration is used, False otherwise
            omega:      relaxation parameter, default: 1
            du_old:     velocity increment of previous time step

        '''
        # TODO: check dw instead of du

        self.use = options['nonlinear']['use_aitken']
        # legacy bool switch compatibility
        if self.use is True:
            self.use = 1
        if self.use is False:
            self.use = 0

        self.picard = (options['nonlinear']['method'] == 'picard')

        self.omega = 1.0
        ''' relaxation parameter '''
        self.du_old = None
        self.du = None
        ''' old residual '''
        pass

    def relax(self, dw, init):
        ''' Compute relaxation parameter, if use > 0 and not first
        iteration (self.dw_old already set).

        Note: keeping du as an instance variable and using assign() on each
        iteration halves the (fairly short) computation time.

        Args:
            dw              current solution increment (dolfin Function)
            init (bool)     True/False switch indicates initialization phase

        Return:
            a1 (float)
        '''
        if self.use == 2 or (self.use and self.picard) or (self.use and init):
            if self.du_old and self.du:
                assign(self.du, dw.sub(0))
                self.omega *= -(
                    np.dot(
                        self.du_old.vector().array(),
                        (self.du.vector() - self.du_old.vector()).array()
                    ) / norm(self.du.vector() - self.du_old.vector(), 'l2')**2
                )
            else:
                # initialize
                (self.du, _) = dw.split(deepcopy=True)
                self.du_old = Function(self.du.function_space())

            assign(self.du_old, self.du)

        return self.omega


class NSSolver:
    ''' Nonlinear solver class for solving a given variational problem
    (object of the NSProblem class).

    Usage:
        ...

    The implemented solvers are:
        - PETSc SNES Newton method
        - 'manual' exact Newton method
        - Quasi-Newton method
        - Picard iteration with Aitken acceleration
    '''
    def __init__(self, nsproblem):
        ''' Initialize Navier-Stokes solver by processing the given problem.

        Args:
            nsproblem (object of NSProblem):      the problem to solve
        '''
        self.options = nsproblem.options
        self.F, self.J = nsproblem.nls_form
        self.Fq, self.Jq = nsproblem.qnls_form
        self.Flin, self.Jlin = nsproblem.ls_form
        self.bcs = nsproblem.bcs
        self.w = nsproblem.w
        self.W = nsproblem.W

        self.energy_form = nsproblem.energy_form

        self.converged_reason = None
        self.it = None
        self._init = None

        self.residual = []
        self.energy = []
        pass

    def solve(self):
        ''' Caller for manual solvers (newtonlike()) and snes().
        Calls of the Newton solvers are preceded by initialization with Picard
        iterations.

        Return:
            self.w (dolfin Function)         final solution
            self.residual (np.ndarray)       residual
        '''
        method = self.options['nonlinear']['method']
        self.it = 0
        if method == 'picard':
            problem = GeneralProblem(self.Flin, self.w, J=self.Jlin,
                                     bcs=self.bcs)
            self.newtonlike(problem)
        else:
            init_steps = self.options['nonlinear']['init_steps']
            # initialize with n=init_steps Picard iterations
            self.initial_solution(init_steps=init_steps)

            if method in ('snes', 'newton'):
                problem = GeneralProblem(self.F, self.w, J=self.J,
                                         bcs=self.bcs)
            elif method == 'qnewton':
                problem = GeneralProblem(self.Fq, self.w, J=self.Jq,
                                         bcs=self.bcs)

            if method == 'snes':
                self.snes(problem)
            elif method in ['newton', 'qnewton']:
                self.newtonlike(problem)

        return self

    def initial_solution(self, init_steps=0):
        ''' Compute initial solution.

        Args:
            init_steps          if 0, compute Stokes solution, otherwise
                                perform n=init_steps Picard iterations
                (default: 0)
        '''
        self._init = True
        lin_problem = GeneralProblem(self.Flin, self.w, J=self.Jlin,
                                     bcs=self.bcs)
        self.newtonlike(lin_problem, init_steps=init_steps)
        self._init = False
        return self

    def newtonlike(self, problem, init_steps=None):
        ''' Newton type solver method for full Newton, quasi-Newton, Picard
        iteration, with optional Aitken delta^2 acceleration.

        If called with init_steps != None, used to compute an initial solution
        for a Newton method.

        Computes until maxit = (init_steps, maxit) or convergence criterion
        reached. Convergence is checked by the method monitor_convergence. The
        AitkenAccelerator is controlled by the class AitkenAccelerator.

        Args:
            init_steps (int, default None):   if set, number of initial steps

        Return:

        '''

        A = PETScMatrix()
        b = PETScVector()
        dw = Function(self.W)

        aitken = AitkenAccelerator(self.options)
        a1 = 1.
        maxit = self.options['nonlinear']['maxit']
        if init_steps is not None:
            maxit = init_steps

        while self.it < maxit:

            # this way, only one call/loop for picard init and newton solver
            # and same AitkenAccelerator can be used!
            # but logic/simplicity not ideal and special treatment for snes
            # necessary
            # solver should be oblivious to problem it is used solving for
            # if init_steps:
            #     self.problem.Jlin(A, self.w.vector())
            #     self.problem.Flin(b, self.w.vector())
            # else:
            #     self.problem.J(A, self.w.vector())
            #     self.problem.F(b, self.w.vector())

            # assemble jacobian and residual
            problem.F(b, self.w.vector())
            problem.J(A, self.w.vector())

            converged, reason = self.monitor_convergence(b, dw, a1)
            if converged:
                print('CONVERGED to {r} in {i} iterations'.
                      format(i=self.it+1, r=reason))
                self.converged_reason = 1
                break

            self.solve_ls(A, dw.vector(), b)

            a1 = aitken.relax(dw, self._init)
            self.w.vector().axpy(-a1, dw.vector())

            self.it += 1

        else:
            if init_steps is None:
                print('NOT CONVERGED, MAX_IT REACHED.')
                self.converged_reason = -5

        return self

    def snes(self, problem):
        ''' Solve the nonlinear problem with SNES with the standard options
        set.
        '''
        # TODO: control over snes options!
        # TODO: check out newtonls (linesearch) variants !
        opt = self.options['nonlinear']

        PETScOptions().set('snes_monitor')
        PETScOptions().set('snes_newtontr')
        PETScOptions().set('snes_converged_reason')
        PETScOptions().set('snes_atol', opt['atol'])
        PETScOptions().set('snes_rtol', opt['rtol'])
        PETScOptions().set('snes_stol', opt['stol'])
        PETScOptions().set('snes_max_it', opt['maxit'])
        PETScOptions().set('ksp_type', 'preonly')
        PETScOptions().set('pc_type', 'lu')
        PETScOptions().set('pc_factor_mat_solver_package', 'mumps')
        solver = PETScSNESSolver()
        solver.init(problem, self.w.vector())
        snes = solver.snes()
        snes.setFromOptions()
        snes.setConvergenceHistory()
        snes.solve(None, as_backend_type(self.w.vector()).vec())
        self.residual = np.hstack((self.residual,
                                   snes.getConvergenceHistory()[0]))
        self.converged_reason = snes.getConvergedReason()

        return self

    def monitor_convergence(self, b, dw, a1):
        ''' Monitor convergence of iteration method. Check if convergence
        criterion is met and store residual in self.residual.

        Args:
            b (PETScVector):    residual vector

        Return:
            converged (bool):   True if converged, False otherwise
        '''
        # TODO: specify this in input file !?
        lp = 'l2'  # l2 vs linf
        residual = norm(b, lp)

        energy = None
        if self.options['nonlinear']['report'] == 2:
            # XXX this should be a temporary hack
            # resid_form = assemble(self.F)
            # energy = np.dot(resid_form.array(), self.w.vector().array())
            energy = assemble(self.energy_form)
            self.energy.append(energy)

        atol = self.options['nonlinear']['atol']
        rtol = self.options['nonlinear']['rtol']
        stol = self.options['nonlinear']['stol']
        init_atol = (self.options['nonlinear']['init_atol'] if 'init_atol' in
                     self.options['nonlinear'] else None)

        dw_norm = norm(dw.vector(), lp)
        init = self.options['nonlinear']['init_steps']

        if self._init and residual <= init_atol:
            converged = True
            reason = 'INIT_ATOL'

        elif residual <= atol:
            converged = True
            reason = 'ATOL'

        # elif residual <= rtol*norm(self.w.vector(), 'l2'):
        #     # FIXME
        #     warnings.warn('In PETSc/SNES, RTOL crit is r0/r_k, not r_k/w_k!')
        #     warnings.warn('My RTOL definition physically meaningless??')
        elif (len(self.residual) >= init + 2 and
              residual <= rtol*self.residual[init + 1]):
            converged = True
            reason = 'RTOL'
            warnings.warn(('RTOL checking starts AFTER init_steps for'
                           'comparability with SNES.'))

        elif (dw_norm <= stol*norm(self.w.vector(), lp) and
              self.it > self.options['nonlinear']['init_steps'] + 1):
            converged = True
            reason = 'STOL'

        else:
            converged = False
            reason = ''

        self.residual.append(residual)

        if self.options['nonlinear']['report']:
            print(('it {it}\t{init}\trelax: {a:.3f}\t residual: {r}'
                  '{s} {e}')
                  .format(it=self.it, r=residual, a=a1,
                          init='i' if self._init else '',
                          e=energy if energy else '',
                          s='\t R(u, u): ' if energy else ''))

        return converged, reason

    def solve_ls(self, A, x, b):
        ''' Solve a linear system
                        A*x = b
            directly with the FEniCS interface.

            Args:
                A           FEniCS/PETSc matrix
                x           DOLFIN Function
                b           rhs PETSc vector
        '''
        solve(A, x, b, self.options['linear']['method'])

        return self

    def reset(self):
        ''' Reset solution w to zero. '''
        self.w.vector().zero()
        self.residual = []
        self._init = None
        self.it = None

        return self


class ReynoldsContinuation:
    ''' Class for applying the Reynolds continuation technique to an instance
    of NSProblem. Starting at a low value, the Reynolds number is gradually
    increased. '''
    def __init__(self, problem, length, u_bulk, Re_sequence=None, Re_start=10,
                 Re_end=None, Re_num=10, init_steps_sequence=None,
                 init_start=2, init_end=10):
        ''' Initialize & set instance variables.

        Args:
            problem     instance of NSProblem
            length      characteristic length for Reynolds number, e.g.
                        hydraulic diameter
            u_bulk      bulk velocity, e.g. for parabolic inlet 2/3*umax

        Optional:
            Re_start (default 10)   if no sequence given, start Reynolds number
            Re_end (default None)   target Re, if None calculate value from
                                    input file
            Re_num (default 10)     number of steps in Reynolds continuation
            Re_sequence             predefined list of Reynolds numbers to
                                    solve for
            init_start (default 2)  number of Picard iterations to start Newton
                                    method for lowest Reynolds number
            init_end (default 10)   number of initial Picard iterations for
                                    target Reynolds number
            init_steps_sequence     predefined list of Picard iterations to
                                    start a Newton method, for each Reynolds
                                    number to be calculated

        '''
        self.problem = problem
        self.length = length
        self.ub = u_bulk
        self.Re_start = Re_start
        self.Re_end = Re_end
        self.Re_num = Re_num
        self.Re_sequence = Re_sequence
        self.init_steps_sequence = init_steps_sequence
        self.init_start = init_start
        self.init_end = init_end
        pass

    def comp_Re_number_from_options(self):
        ''' Calculates Reynolds number from u_bulk, mu, rho, characteristic
        length.
        '''
        L = self.length
        rho = self.problem.options['rho']
        mu = self.problem.options['mu']
        ub = self.ub
        self.Re_end = rho*L*ub/mu

        return self

    def comp_mu_from_Re(self, Re):
        ''' Compute viscosity mu from a given Reynolds number with L, U,
        specified and rho from parameters.

        Args:
            Re  Reynolds number

        Return:
            mu  viscosity
        '''
        L = self.length
        rho = self.problem.options['rho']
        ub = self.ub

        mu = rho*L*ub/Re

        return mu

    def create_Re_sequence(self):
        ''' Calculate Reynolds number sequence. Predefined spacing is x^1.5,
        which works well for the backward-facing step case and Re = 800.
        '''
        if not self.Re_end:
            self.comp_Re_number_from_options()

        x = np.linspace(self.Re_start**1.5, self.Re_end**1.5, self.Re_num)
        self.Re_sequence = x**(1./1.5)

        return self

    def create_init_steps_sequence(self):
        ''' Calculate init step number sequence, log spaced. '''
        x = np.logspace(np.log10(self.init_start),
                        np.log10(self.init_end), self.Re_num)
        self.init_steps_sequence = x.round().astype(int)

        return self

    def solve(self):
        ''' Loop over Reynolds numbers and solve problem f.e. Re. Stop in case
        of non-convergence of a solve.
        '''
        if self.Re_sequence is None:
            self.create_Re_sequence()
        else:
            self.Re_num = len(self.Re_sequence)

        if self.init_steps_sequence is None:
            self.create_init_steps_sequence()

        for i, (isteps, Re) in enumerate(zip(self.init_steps_sequence,
                                             self.Re_sequence)):
            print('{i}/{N}\t isteps: {s}\t Re: {Re}'
                  .format(i=i+1, N=self.Re_num, s=isteps, Re=Re))
            self.problem.options['mu'] = self.comp_mu_from_Re(Re)
            self.problem.options['nonlinear']['init_steps'] = isteps
            if i == 0:
                self.problem.init()
            else:
                self.problem.variational_form()

            self.sol = NSSolver(self.problem)
            self.sol.solve()
            if self.sol.converged_reason < 0:
                raise Exception('Diverged in step {0} for Re = {1}'
                                .format(i, Re))
                break

        self.w = self.sol.w

        return self
