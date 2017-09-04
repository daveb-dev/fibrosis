''' Steady Navier-Stokes estimator module

Author: David Nolte (dnolte@dim.uchile.cl)
Date: 20-09-2016
'''

# TODO: Throw away bctype option??

from dolfin import *
from functions import inout
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brute
from nsproblem import NSProblem
from nssolver import NSSolver
import warnings


class NSEstimator:
    ''' Estimate the coefficients of Navier-Slip boundary conditions using the
    NSSolver class.

    The Navier-Slip BCs need to be specified for every boundary segment
    (defined as gmsh physical group) in the estimator input file with 'preset:
        'navierslip''. Only the Nitsche method is currently supported.

    Note: in the iteration process, the previous solution is used as initial
        solution for each solve.

    Currently implemented optimization methods:
        - scipy.optimize.minimize
    '''
    def __init__(self, opt_ref, opt_est):
        ''' Initialize Navier-Slip Estimator class.

        Args:
            opt_ref         path to yaml file with settings for reference case
            opt_est         path to yaml file with settings for estimation
        '''
        self.optfile_ref = opt_ref
        self.optfile_est = opt_est
        self.pb_est = None
        self.pb_ref = None

        self.options = inout.read_parameters(opt_est)

        self.uref = None
        self.pref = None
        self.pref_meas = None
        self.uref_meas = None
        self.u_meas = None
        # TODO: in the end self.u_meas is the LAST iterate, not the optimal
        #       one!!
        self.u_opt = None
        self.p_opt = None

        self._x0 = None
        self._xfun = None
        self._bounds = None
        self.xlegend = None

        self.x_opt = None
        self.f_opt = None
        self.fval = []
        self.x = []
        self.beta = []
        self.result = None
        self.p_stei = None
        self.BO = None

        self._end = False

        self.init_problems()

        pass

    def init_problems(self):
        ''' Create reference problem and estimator problem.
        Automatically called by self.__init__(). Decoupled by the actual
        pb.init() calls at the beginning of self.estimate(), so parameters can
        be changed more easily between calls.
        '''
        self.pb_ref = NSProblem(self.optfile_ref)
        self.pb_est = NSProblem(self.optfile_est)
        return self

    def _init_measurement(self):
        ''' Initialize measurement function space and functions.'''
        opt_meas = self.options['estimation']['measurement']
        mesh, _, bnds = inout.read_mesh(opt_meas['mesh'])
        self.bnds_meas = bnds  # needed for pressure_drop
        if opt_meas['elements'] == 'P1':
            deg = 1
        elif opt_meas['elements'] == 'P2':
            deg = 2
        else:
            raise ValueError('Element type unknown. Available options: P1, P2')
        V = FunctionSpace(mesh,
                          VectorElement('Lagrange', mesh.ufl_cell(), deg))
        Q = FunctionSpace(mesh, FiniteElement('Lagrange', mesh.ufl_cell(), 1))
        self.pref_meas = Function(Q)
        self.uref_meas = Function(V)
        self.u_meas = Function(V)

        return self

    def _interp_measurement(self, u, ref=False):
        ''' Interpolate velocity field to measurement function space.

        Args:
            u       velocity field to interpolate
            ref     reference solution flag
        '''
        if ref:
            if not self.uref_meas:
                raise Exception('uref_meas is None. Call _init_measurement '
                                'first!')
            uinp = self.uref_meas
        else:
            if not self.u_meas:
                raise Exception('u_meas is None. Call _init_measurement '
                                'first!')
            uinp = self.u_meas

        LI = LagrangeInterpolator()
        LI.interpolate(uinp, u)

        return self

    def add_gaussian_noise(self, u, scal_umax):
        ''' Add Gaussian noise to a velocity field u, with amplitude
        scal_umax*umax.

        Args:
            u (dolfin function)       velocity field
            scal_umax                 scaling factor for noise amplitude
        '''
        assert self.u_meas, 'self.u_meas doesn\'t exist yet!'
        if 'random_seed' in self.pb_est.options['estimation']:
            np.random.seed(self.pb_est.options['estimation']['random_seed'])

        dim = u.vector().size()
        umax = abs(u.vector().array()).max()
        noise = np.random.normal(0., scal_umax*umax, dim)
        noise_fun = Function(u.function_space())
        noise_fun.vector()[:] = noise
        u.vector().axpy(1.0, noise_fun.vector())
        pass

    def _update_parameters_inflow(self, x):
        ''' Update coefficient of inflow profile. Modifies Expressions stored
        in NSProblem.bcs.
        Procedure:  1) Find inlet BC in list of strong Dirichlet BCs.
                    2) Update value at position. According to settings, change
                        U only or U and R.


        # TODO: PROBLEM IS, R = R + dR WILL BE OVERWRITTEN !
        # SOLVED?: SPLIT options dict and bc_lst which is modified by BCs in
            class NSProblem

            ATTENTION: This is quite fragile and depends on the fact that the
            boundary conditions are processed in order ...

        Args:
            x       parameter: max inflow velocity x[0], dR x[1] (if set)
        '''

        bc_lst = self.pb_est.bc_lst
        param = self.options['estimation']['parameters']['inflow']
        assert param['use']

        val = None

        count_dbc = 0
        for i_bc, bc in enumerate(bc_lst):
            # count_dbc dirichlet BCs before inlet
            if 'preset' in bc and bc['preset'] == 'inlet':
                val = bc['value']
                bid = bc['id']
                break
            elif 'method' in bc and bc['method'] == 'essential':
                count_dbc += 1
        else:
            raise KeyError('Inlet BC not found in input file.')

        assert bc_lst[i_bc]['id'] == bid
        assert bc_lst[i_bc]['preset'] == 'inlet'

        val.U = x[0]

        if param['use'] == 2:
            # use_slip does not matter. dR will always be in second position in
            # parameters vector. Maybe necessary TODO this in the future.
            warnings.warn('Take care. inlet>value>R needs to be INNER radius.')
            R_inlet = self.options['boundary_conditions'][i_bc]['value']['R']
            val.R = x[1] + R_inlet

        V = self.pb_est.W.sub(0)
        if self.pb_est.is_enriched(V):
            val = project(val, V.collapse(), solver_type='lu')

        self.pb_est.bcs[count_dbc].set_value(val)

        return self

    def _update_parameters_navierslip(self, x):
        ''' Update coefficients of Navier-Slip BCs.

        Modifies Expressions stored in of NSProblem.bcs_navierslip.  Selects
        the gamma prefactor or dR according to the 'parameters' setting in the
        options file.

        Args:
            x       parameter vector
        '''
        param = self.options['estimation']['parameters']['navierslip']
        assert param['use']

        if 'boundary_id' in self.options['estimation']:
            boundary_selection = self.options['estimation']['boundary_id']
            if type(boundary_selection) is int:
                boundary_selection = [boundary_selection]
        else:
            boundary_selection = [0]
        if boundary_selection[0] or len(self.pb_est.bcs_navierslip) > 1:
            raise NotImplementedError('Only one parameter per boundary '
                                      'supported currently.')
        if not self.pb_est.bcs_navierslip:
            raise Exception('No Navier-slip boundary found.')

        # find position of Navier-slip coefficient in parameter vector
        index = self.xlegend.index('navierslip')
        val = x[index]
        #  for val, bc in zip(, self.pb_est.bcs_navierslip):
        #      if bc[0] in boundary_selection or boundary_selection == [0]:

        for bc in self.pb_est.bcs_navierslip:
            if param['use'] == 1:
                bc[1].a = val
            elif param['use'] == 2:
                bc[1].dR = val

        return self

    def _update_parameters_nitsche(self, beta):
        ''' Update coefficient of the Nitsche boundary conditions.

        NOTE: Careful, this updates the betas of ALL Nitsche BCs!!!

        Args:
            beta       new value for beta
        '''
        self.pb_est.options['nitsche']['beta1'] = beta

        raise Exception('Nitsche Optimization not supported in the current '
                        'version')

        return self

    def _update_parameters_transpiration(self, x):
        ''' Update coefficients of transpiration BC.
        Modifies Expression stored in NSProblem.bcs_transpiration.

        Args:
            x       parameters vector
        '''
        param = self.options['estimation']['parameters']['transpiration']
        assert param['use']

        if 'boundary_id' in self.options['estimation']:
            boundary_selection = self.options['estimation']['boundary_id']
            if type(boundary_selection) is int:
                boundary_selection = [boundary_selection]
        else:
            boundary_selection = [0]
        if boundary_selection[0] or len(self.pb_est.bcs_transpiration) > 1:
            raise NotImplementedError('Only one parameter per boundary '
                                      'supported currently.')
        if not self.pb_est.bcs_transpiration:
            raise Exception('No transpiration boundary found.')

        # find correct index in parameters array
        index = self.xlegend.index('transpiration')
        val = x[index]

        for bc in self.pb_est.bcs_transpiration:
            assert self.pb_est.is_Constant(bc[1])
            bc[1].assign(val)

        return self

    def _update_parameters(self, x):
        ''' Update coefficients of BCs. Depending on the problem:
            no-slip:     bctype == 0.   -> inflow (U)
            navier-slip: bctype == 1.   -> any combination of inflow(U, dR),
                                    dR or gamma, beta(Nitsche/Transpiration)

        Args:
            x        parameter

        '''
        param = self.options['estimation']['parameters']

        if param['inflow']['use']:
            self._update_parameters_inflow(x)

        if param['navierslip']['use']:
            # navier-slip
            self._update_parameters_navierslip(x)

        if param['transpiration']['use']:
            # transpiration
            self._update_parameters_transpiration(x)

        self.pb_est.variational_form()

        return self

    def _apply_xfun(self, x):
        ''' Apply 'xfun' to parameters x.

        xfun == 0: linear, y = x
        xfun == 1: exponential, y = 2**x
        xfun == 2: tanh, y = a + b*0.5*(np.tanh(x) + 1)

        Args:
            x (ndarray)   parameter

        Returns:
            y (ndarray)   result of xfun(x)
        '''
        # check for bruteforce 1 parameter corner case
        if type(x) in (np.ndarray, np.float64) and not x.shape:
            x = np.array([x])
        yi = []
        for i, (xi, fi, bi) in enumerate(zip(x, self._xfun, self._bounds)):
            if fi == 0:
                # linear
                yi.append(xi)
            if fi == 1:
                # exponential
                yi.append(2**xi)
            if fi == 2:
                # tanh
                yi.append(self.tanh_xfun(xi, bi))

        return np.array(yi)

    def _apply_inv_xfun(self, x):
        ''' Apply inverse xfun to initial parameters.
        See _apply_xfun().

        Args:
            x       parameters

        Returns
            y       inv_xfun(x)
        '''
        yi = []
        for i, (xi, fi, bi) in enumerate(zip(x, self._xfun, self._bounds)):
            if fi == 0:
                # linear
                yi.append(xi)
            if fi == 1:
                # exponential
                yi.append(np.log2(xi))
            if fi == 2:
                # tanh
                yi.append(self.inv_tanh_xfun(xi, bi))

        return np.array(yi)

    def _tikhonov_regularization(self, val):
        ''' Tikhonov regularization.

        Returns:
            val     contribution to fval
        '''
        raise NotImplementedError()
        #  if opt_est['xfun'] == 1:
        #      val0 = 2**self.x0
        #  elif opt_est['xfun'] == 2:
        #      val0 = abs(self.x0)
        #  elif opt_est['xfun'] == 3:
        #      val0 = self.x0**2
        #  else:
        #      val0 = self.x0
        #  tikh = opt_est['tikhonov']*np.linalg.norm(val0 - val)**2
        #  if opt_est['error'] == 'rel':
        #      tikh /= np.linalg.norm(val0)**2
        return val

    def _compute_error(self):
        ''' Compute the L2 error of the calculated velocity field w.r.t to the
        measurement.

        Returns:
            fval    error
        '''
        u, _ = self.pb_est.w.split(deepcopy=True)
        self._interp_measurement(u)  # -> stored to self.u_meas

        fval = norm(self.u_meas.vector() - self.uref_meas.vector(), 'l2')

        if self.options['estimation']['error'] == 'rel':
            fval /= norm(self.uref_meas.vector(), 'l2')

        return fval

    def _solve(self, x):
        ''' Solve function called by optimization method.

        # TODO: clean up the mess (Tikhonov??)

        Args:
            x                   estimation parameter

        Returns:
            fval                value to be minimized: error wrt measurement
        '''
        opt_est = self.options['estimation']

        val = self._apply_xfun(x)

        self._update_parameters(val)
        solver = NSSolver(self.pb_est)
        solver.solve()

        assert id(self.pb_est.w) == id(solver.w) and self.pb_est.w == solver.w

        fval = self._compute_error()

        if opt_est['tikhonov']:
            fval += self._tikhonov_regularization(val)

        if not self._end:
            print('Parameters:\t {0}'.format(str(val)))
            print('Fval:\t\t {0}'.format(fval))
            self.fval.append(fval)
            self.x.append(x)

        return fval

    def _setup_bruteforce(self):
        ''' Setup bruteforce arguments. If as_slices is set, make slices
        according to Npts (int: uniform, or list) and bounds. Otherwise assure
        that Npts is an integer.

        Returns:
            Npts (int)      Number of points (no slices)
            bounds        tuple of slices or list of limits for each
                            parameter
        '''
        opt_est = self.options['estimation']
        bounds = self._bounds

        Npts = opt_est['bruteforce']['numpts']
        if opt_est['bruteforce']['as_slice']:
            if not type(Npts) is list:
                Npts = [Npts]*len(bounds)
            slices = []
            for (n, bnd) in zip(Npts, bounds):
                step = (bnd[1] - bnd[0])/(n - 1)
                slices.append(slice(bnd[0], bnd[1] + 1.e-10, step))
            bounds = tuple(slices)

        else:
            assert type(Npts) is int, (
                'If [a, b] ranges are given, numpts must be int. Use '
                'slices for nonuniform grids.')

        return Npts, bounds

    def _parse_parameters(self):
        ''' Cast parameters into the required form.
        Process inflow (U, (dR)), navierslip (gamma or dR), transpiration coef.
        The inflow dR can be taken from the navierslip estimate, if dR is
        chosen to be optimized in the navierslip section, via the use_slip
        switch.

        For all optimization parameters, the initial value x0, the parameter
        function xfun, and the limits (if any), are added to the instance
        variables:
            self._x0, self._xfun, self._bounds

        The initial values x0 are expected to be the 'true' physical values,
        BEFORE re-parametrization.  The inverse of 'xfun'
        (self._apply_inv_xfun) is applied on x0 in the end in order to get the
        correct values.

        The order is:
            [u_inflow, dR_inflow, navierslip, transpiration]

        '''
        param = self.options['estimation']['parameters']

        # create start vector, x0
        self.xlegend = []
        self._x0 = []
        self._xfun = []
        self._bounds = []

        if param['inflow']['use']:
            self._xfun.append(param['inflow']['velocity']['xfun'])
            self._x0.append(param['inflow']['velocity']['x0'])
            self._bounds.append(param['inflow']['velocity']['bounds'])
            self.xlegend.append('Uin')

            if (param['inflow']['use'] == 2 and
                    param['inflow']['dR']['use_slip'] == 0):
                self._x0.append(param['inflow']['dR']['x0'])
                self._xfun.append(param['inflow']['dR']['xfun'])
                self._bounds.append(param['inflow']['dR']['bounds'])
                self.xlegend.append('dR_in')
            elif (param['inflow']['dR']['use_slip'] == 1 and not
                    param['navierslip']['use'] == 2):
                raise Exception('Inflow dR to be taken from Navier-Slip dR'
                                ' but dR estimation via Navier-Slip set!')

        estim_boundaries = ['navierslip', 'transpiration']
        for bnd in estim_boundaries:
            if param[bnd]['use']:
                self._x0.append(param[bnd]['x0'])
                self._xfun.append(param[bnd]['xfun'])
                self._bounds.append(param[bnd]['bounds'])
                self.xlegend.append(bnd)

        self._x0 = self._apply_inv_xfun(self._x0)
        return self

    def gpyopt_optimization(self):
        ''' OPtimization using GPyOpt.

        Returns
            results     x_opt, f_opt dict
        '''
        import GPyOpt
        bounds = self.options['estimation']['gpyopt']['bounds']
        if not bounds:
            gpbounds = None
        else:
            gpbounds = [
                {'name': 'x{0}'.format(i), 'type': 'continuous', 'domain':
                 tuple(gpbnd)} for (i, gpbnd) in enumerate(bounds)]

        if ('x0' in self.options['estimation']['gpyopt'] and
                type(self.options['estimation']['gpyopt']['x0']) is list):
            Xinit = np.array(self.options['estimation']['gpyopt']['x0'])
        else:
            Xinit = None

        acq_type = self.options['estimation']['gpyopt']['acq_type']
        model_type = self.options['estimation']['gpyopt']['model_type']

        myBO = GPyOpt.methods.BayesianOptimization(
            f=self._solve,
            domain=gpbounds,
            acquisition_type=acq_type,
            model_type=model_type,
            X=Xinit
        )

        max_iter = self.options['estimation']['gpyopt']['max_iter']
        max_time = self.options['estimation']['gpyopt']['max_time']
        eps = 1e-6
        myBO.run_optimization(max_iter, max_time, eps)

        plt.ion()
        myBO.plot_acquisition()

        self.BO = myBO
        result = {'x': myBO.x_opt, 'f': myBO.fx_opt}
        self.x = np.array(self.x).squeeze()

        return result

    def measurement(self):
        ''' Makes measurement: first, compute reference solution, then
        interpolate to measurement mesh and add noise. '''

        self.reference_solution()
        self._init_measurement()
        self._interp_measurement(self.uref, ref=True)

        noise_intensity = self.options['estimation']['noise']
        if noise_intensity:
            self.add_gaussian_noise(self.uref_meas, noise_intensity)

        return self

    def reference_solution(self):
        ''' Compute reference solution and produce measurement (u_meas). '''

        self.pb_ref.init()
        sol = NSSolver(self.pb_ref)
        sol.solve()
        self.uref, self.pref = sol.w.split(deepcopy=True)

        return self

    def estimate(self):
        '''Estimate parameters of Navier-Slip BC.
        Setup problem from yaml file and call optimization method with set of
        initial values.
        Note: NSSolver initialization takes 1.4us, so no reason to setup
            beforehand.

        TODO NOTE: included now beta optimization via switch in yaml file.

        Args:
            x0 (optional)       initial values

        '''
        opt_est = self.options['estimation']

        self._parse_parameters()

        self.measurement()
        self.pb_est.init()

        method = opt_est['method']
        if method == 'Powell':
            result = minimize(self._solve, self._x0, method='Powell')
            self.x_opt = result['x']
            self.f_opt = result['fun']

        elif method == 'Nelder-Mead':
            result = minimize(self._solve, self._x0, method='Nelder-Mead')
            #                options={'disp': True,
            #                         'xtol': 1e-2, 'ftol': 1e-2})
            self.x_opt = result['x']
            self.f_opt = result['fun']

        elif method == 'BFGS':
            result = minimize(self._solve, self._x0, method='BFGS',
                              tol=opt_est['bfgs']['tol'])
            #                  options={
            #                      'disp': True, 'gtol': 1e-5, 'eps': 1e-3
            #                  })
            self.x_opt = result['x']
            self.f_opt = result['fun']

        elif method == 'bruteforce':
            Npts, bfbounds = self._setup_bruteforce()
            result = brute(self._solve, bfbounds, Ns=Npts, disp=True,
                           finish=None, full_output=True)
            # finish (default) = scipy.optimize.fmin to polish result
            self.x_opt = result[0]
            self.f_opt = result[1]

        elif method == 'gpyopt':
            raise Exception('GPyOpt dropped. Adapt...')
            result = self.gpyopt_optimization()
            self.x_opt = result['x']
            self.f_opt = result['f']

        # optimization done.
        self._end = True

        self.fval = np.array(self.fval)
        self.x = np.array(self.x)

        print(result)
        self.result = result

        return self

    def solve_opt(self, x=None, init=False):
        ''' Solve with the optimal parameters.

        Args:
            x (optional, numpy.ndarray)   parameters; if not given, use x_opt
            init (optional, bool)         reinitialize solution w
        '''
        if x is None:
            x = self.x_opt
        #  else:
        #      x = self._apply_inv_xfun(x)
        if init:
            self.pb_est.w.vector().zero()
            print('zeroed')
        self._solve(x)
        self.u_opt, self.p_opt = self.pb_est.w.split(deepcopy=True)
        self.u_meas_opt = self.u_meas
        return self

    def get_radius_at_vert_boundary(self, bnds, bid):
        ''' Get the radius of a vertical boundary patch.

        Args:
            bnds    boundary domain object
            bid     boundary id

        Returns:
            radius
        '''
        It_facet = SubsetIterator(bnds, bid)
        ycoord = []
        for c in It_facet:
            for v in vertices(c):
                ycoord.append(v.point().y())

        ycoord = np.array(ycoord)
        if np.allclose(ycoord.min(), 0) or np.allclose(ycoord.max(),
                                                       -ycoord.min()):
            # symmetric or  full (-R, R)
            radius = ycoord.max()
        else:
            warnings.warn('Pressure_drop: careful, geometry not symmetric! '
                          'ymin = {0}, ymax = {1}'.format(ycoord.min(),
                                                          ycoord.max()))
            radius = 0.5*(ycoord1.max() - ycoord1.min())

        return radius

    def pressure_drop(self, p, sin=1, sout=2):
        ''' Calculate pressure drop for optimized NSE solution or reference
        pressure on the respective meshes, between two boundaries sin, sout.
        The pressure is integrated over the boundaries and devided by the
        respective measure (integral mean), then substracted.

        The function detects automatically if the given pressure field p is
        defined on a) the reference mesh, b) the estimation mesh, c) the
        measurement mesh, and the boundary FacetFunction is chosen
        appriopriately.

        Args:
            p       pressure field
            sin     index of inlet boundary
            sout    index of outlet boundary
        '''
        # detect reference or estimator problem
        mesh = p.function_space().mesh()
        if mesh.id() == self.pref.function_space().mesh().id():
            # reference case
            bnds = self.pb_ref.bnds
        elif (mesh.id() ==
              self.pb_est.w.split(deepcopy=True)[1].function_space().
              mesh().id()):
            bnds = self.pb_est.bnds
        elif mesh.id() == self.pref_meas.function_space().mesh().id():
            bnds = self.bnds_meas
        else:
            raise Exception('p not identified.')

        ds = Measure('ds', domain=mesh, subdomain_data=bnds)

        measure_sin = Constant(self.get_radius_at_vert_boundary(bnds, sin))
        measure_sout = Constant(self.get_radius_at_vert_boundary(bnds, sout))
        #  print('measure_sin:  {0}'.format(measure_sin.values()[0]))
        #  print('measure_sout: {0}'.format(measure_sin.values()[0]))

        dP = assemble(p/measure_sin*ds(sin) - p/measure_sout*ds(sout))

        return dP

    def direct_estimator(self, method='STEint', return_pressure=False):
        ''' Compute "standalone" pressure estimate; caller function to
        encompass Navier-slip optimization via self.estimation().

        Args:
            method      pressure estimation method: STE, STEint, PPE

        Returns:
            dP          estimated pressure drop
        '''
        if not self.uref_meas:
            self.measurement()
        if not self.pb_est.w:
            self.pb_est.init()

        fun = getattr(self, method)
        dP, p_est = fun()

        if return_pressure:
            ret = (dP, p_est)
        else:
            ret = dP
        return ret

    def PPE(self, sin=1, sout=2):
        ''' Compute PPE pressure approximation and pressure drop.

        Args (optional):
            sin         inlet boundary id
            sout        outlet boundary id

        Returns:
            dP
        '''
        assert self.uref_meas, 'Reference measurement does not exist.'
        rho = self.pb_ref.options['rho']

        mesh = self.uref_meas.function_space().mesh()

        E1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
        P1 = FunctionSpace(mesh, E1)

        p = TrialFunction(P1)
        q = TestFunction(P1)

        bc = DirichletBC(P1, Constant(0.), self.bnds_meas, sout)

        u0 = self.uref_meas

        a = inner(grad(p), grad(q))*dx
        L = - rho*inner(grad(u0)*u0, grad(q))*dx

        A, b = assemble_system(a, L, bc)

        p_est = Function(P1)
        solve(A, p_est.vector(), b, 'mumps')

        self.p_est = p_est

        dP = self.pressure_drop(p_est)
        return dP, p_est

    def STE(self, sin=1, sout=2):
        ''' Compute STE pressure approximation and pressure drop.

        Args (optional):
            sin         inlet boundary id
            sout        outlet boundary id

        Returns:
           dP
        '''
        assert self.uref_meas, 'Reference measurement does not exist.'
        rho = self.pb_ref.options['rho']

        ndim = self.pb_ref.ndim

        mesh = self.uref_meas.function_space().mesh()

        P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
        B = FiniteElement('Bubble', mesh.ufl_cell(), 1 + ndim)
        W = FunctionSpace(mesh, MixedElement(ndim*[P1 + B])*P1)

        (w, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)

        zero = Constant((0,)*ndim)
        noslip = project(zero, W.sub(0).collapse())
        bc = DirichletBC(W.sub(0), noslip, 'on_boundary')

        u0 = self.uref_meas

        a = inner(grad(w), grad(v))*dx - p*div(v)*dx + div(w)*q*dx
        L = - rho*inner(grad(u0)*u0, v)*dx

        # A = assemble(a)
        # b = assemble(L)
        # bc.apply(A, b)
        A, b = assemble_system(a, L, bc)

        w1 = Function(W)
        solve(A, w1.vector(), b, 'mumps')

        _, p_est = w1.split(deepcopy=True)
        self.p_est = p_est

        dP = self.pressure_drop(p_est)
        return dP, p_est

    def STEint(self, sin=1, sout=2):
        ''' Compute STEint pressure approximation and pressure drop.

        Args (optional):
            sin         inlet boundary id
            sout        outlet boundary id

        Returns:
           dP
        '''
        assert self.uref_meas, 'Reference measurement does not exist.'
        mu = self.pb_ref.options['mu']
        rho = self.pb_ref.options['rho']

        ndim = self.pb_ref.ndim

        mesh = self.uref_meas.function_space().mesh()

        P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
        B = FiniteElement('Bubble', mesh.ufl_cell(), 1 + ndim)
        W = FunctionSpace(mesh, MixedElement(ndim*[P1 + B])*P1)

        (w, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)

        zero = Constant((0,)*ndim)
        noslip = project(zero, W.sub(0).collapse())
        bc = DirichletBC(W.sub(0), noslip, 'on_boundary')

        u0 = self.uref_meas

        a = inner(grad(w), grad(v))*dx - p*div(v)*dx + div(w)*q*dx
        L = - mu*inner(grad(u0), grad(v))*dx + rho*inner(grad(v)*u0, u0)*dx

        # A = assemble(a)
        # b = assemble(L)
        # bc.apply(A, b)
        A, b = assemble_system(a, L, bc)

        w1 = Function(W)
        solve(A, w1.vector(), b, 'mumps')

        _, p_est = w1.split(deepcopy=True)
        self.p_est = p_est

        dP = self.pressure_drop(p_est)
        return dP, p_est

    def gamma(self, x, R_i=0.95, R_o=1.0):
        ''' Utility function for calculating Navier-slip Gamma from the
        optimization parameters. First the poiseuille base gamma is computed,
        then the gamma based on the estimation parameters.
        Since the Navier-slip BC is defined via x*gamma_pois, where gamma_pois
        is the Poiseuille gamma obtained from the physical parameters and only
        the proportionality factor x is set, both gammas are returned for
        comparability.

        Args:
            xi      parameter(s)
            R_i     Poiseuille gamma inner radius
            R_o     Poiseuille gamma outer radius

        Returns:
            gamma_pois      Poiseuille gamma
            gamma_opt           optimized gamma
        '''

        use = self.options['estimation']['parameters']['navierslip']['use']
        mu = self.options['mu']

        gamma_pois = 2.*mu*R_i/(R_i**2 - R_o**2)

        if use == 1:
            # xi*gamma  optimization
            gamma_opt = x*2.*mu*R_i/(R_i**2 - R_o**2)
        elif use == 2:
            # dR optimization
            R_o = R_i + x
            gamma_opt = 2.*mu*R_i/(R_i**2 - R_o**2)

        return gamma_pois, gamma_opt

    def tanh_xfun(self, x, bounds):
        ''' Compute tanh(x) function.

        Args:
            x           evaluation location
            bounds      tuple with (lower, upper) limits

        Return:
            beta
        '''
        return bounds[0] + bounds[1]*0.5*(np.tanh(x) + 1)

    def inv_tanh_xfun(self, x, bounds):
        ''' Compute inverse of tanh(x) function.

        Args:
            x           evaluation location
            bounds      tuple with (lower, upper) limits

        Return:
            beta
        '''
        return np.arctanh(2./bounds[1]*(x - bounds[0]) - 1.)
