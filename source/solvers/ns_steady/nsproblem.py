from dolfin import *
import dolfin
import ufl
from functions import geom
import numpy as np
from functions import inout
from functions import utils
import warnings
import copy


class NSProblem:
    ''' Class that sets up the Navier-Stokes problem to be solved.
    '''
    # TODO: could inherit from a ProblemBaseClass that has all the  check and
    # help functions
    # TODO: simplify boundary conditions. Use-cases are basically known, assume
    #   that only the 'preset' BCs will be used in the feature and remove the
    #   switch!
    # TODO: Copy the boundary conditions dict instead of changing the options
    #       dict?
    def __init__(self, opt_file=None):
        ''' Initialize Navier-Stokes problem. Initalizes all instance variables
        with None and loads parameter file.

        Args:
            opt_file (str):     path to YAML input file

        Attributes:
            self.options:   options dictionary
            self.nls_form:  list variational forms of residual F and jacobian J
                            of the nonlinear problem
            self.ls_form:   list variational forms of residual F and jacobian J
                            of the bilinear (linearized) problem
            self.qnls_form: ls_form + newton linearization terms (qnewton)
            self.bcs:       list of boundary conditions
            self.W:         function space
            self.w:         solution Function(W)
            self.bnds       boundary MeshFunction
        '''

        self.options = None
        if opt_file:
            self.get_parameters(opt_file)

        self.mesh = None
        self.bnds = None
        self.sd = None

        self.bc_lst = None

        self.ls_form = None
        self.nls_form = None
        self.qnls_form = None
        self.bcs = []
        self.bcs_neumann = []
        self.bcs_nitsche = []
        self.bcs_navierslip = []
        self.bcs_transpiration = []

        self.w = None
        self.W = None

        self._bid_outlet = None

    def init(self):
        ''' Initialize problem:
            1. function spaces
            2. boundary conditions
            3. variational forms
        '''
        if self.options:
            self.init_mesh()
            self.mixed_functionspace()
            self.boundary_conditions()
            self.variational_form()
        else:
            raise Exception('Options not set. call get_parameters(optfile) '
                            'first!')

        if ('load_data' in self.options and 'data_file' in self.options and
                self.options['load_data'] and self.options['data_file']):
            # read data from HDF5 File
            try:
                self.load_HDF5()
            except:
                warnings.warn('Data could not be loaded: file {0} does not '
                              'exist. Continuing...'.format(
                                  self.options['data_file']))
        return self

    def save_HDF5(self, filepath=None):
        ''' Save solution to a HDF5 file, specified by the data_file key in the
        options dict.

        Args:
            filepath (str)    path to hdf5 file to be written
        '''
        if not filepath:
            filepath = self.options['data_file']

        print('writing solution function to file: {0}'.format(filepath))
        inout.write_HDF5_data(filepath, self.mesh.mpi_comm(),
                              self.w, '/w')
        pass

    def load_HDF5(self):
        ''' Load function from a HDF5 file, specified by the data_file key in
        the options dict. '''
        print('loading solution function from file: {0}'.format(
            self.options['data_file']))
        inout.read_HDF5_data(self.options['data_file'], self.mesh.mpi_comm(),
                             self.w, '/w')
        pass

    def get_parameters(self, opt_file):
        ''' Reads parameters from YAML input file into options dictionary

        Args:
            opt_file (str):     path to YAML file
        '''
        self.options = inout.read_parameters(opt_file)
        return self

    def init_mesh(self):
        ''' Read and store mesh, subdomains and boundary information. '''
        self.mesh, self.subdomains, self.bnds = \
            inout.read_mesh(utils.prep_mesh(self.options['mesh']))
        self.ndim = self.mesh.topology().dim()
        return self

    def mixed_functionspace(self):
        ''' Create mixed function space
                W = V x Q
        on given mesh and define the solution function, w \in W.

        Implemented options, set by options['elements']:
            'TH':    Taylor-Hood P2/P1
            'Mini':  Mini P1+Bubble/P1
            'P1':    P1/P1

        '''
        elem = self.options['elements']

        if '2016.1' in dolfin.__version__:
            if elem == 'TH':
                P2 = VectorElement('Lagrange', self.mesh.ufl_cell(), 2)
                P1 = FiniteElement('Lagrange', self.mesh.ufl_cell(), 1)
                W = FunctionSpace(self.mesh, P2 * P1)
            elif elem == 'Mini':
                #  P1 = FunctionSpace(self.mesh, 'CG', 1)
                #  B = FunctionSpace(self.mesh, 'Bubble', 1 + self.ndim)
                #  V = MixedFunctionSpace(self.ndim*[P1 + B])

                P1 = FiniteElement('Lagrange', self.mesh.ufl_cell(), 1)
                B = FiniteElement('Bubble', self.mesh.ufl_cell(),
                                  1 + self.ndim)
                W = FunctionSpace(self.mesh,
                                  MixedElement(self.ndim*[P1 + B])*P1)

            elif elem == 'P1':
                V1 = VectorElement('Lagrange', self.mesh.ufl_cell(), 1)
                P1 = FiniteElement('Lagrange', self.mesh.ufl_cell(), 1)
                W = FunctionSpace(self.mesh, V1 * P1)

        elif dolfin.__version__ == '1.6.0':
            warnings.warn('Using old FEniCS version 1.6. Prefer v2016.x.')
            if elem == 'TH':
                V = VectorFunctionSpace(self.mesh, 'CG', 2)
            elif elem == 'Mini':
                P1 = FunctionSpace(self.mesh, 'CG', 1)
                B = FunctionSpace(self.mesh, 'Bubble', 1 + self.ndim)
                V = MixedFunctionSpace(self.ndim*[P1 + B])
            elif elem == 'P1':
                V = VectorFunctionSpace(self.mesh, 'CG', 1)
            Q = FunctionSpace(self.mesh, 'CG', 1)
            W = V*Q
        else:
            raise Exception('dolfin version not supported ({0})'.format(
                dolfin.__version__))

        # self.W = W
        w = Function(W)
        w.vector().zero()

        self.w = w
        self.W = W

        return self

    def variational_form(self):
        ''' Caller for variational form functions.
        Build forms of bilinear (i.e., linearized) and nonlinear problem.
        Store in self.nls_form (F, J) and self.ls_form (F, J). ls_form is
        extended by Newton linearization terms (for quasi-Newton methods) and
        stored in self.qnls_form.
        '''

        if not self.W:
            self.mixed_functionspace()
        assert self.W and self.w, ('Function space W and function w not'
                                   'initialized.')

        self.bilinear_form()

        self.nonlinear_form()

        return self

    def nonlinear_form(self):
        ''' Build residual and residual Jacobian of the nonlinear variational
        form.

        Attributes:
            nls_form        tuple (F, J)
        '''
        mu = Constant(self.options['mu'])
        rho = Constant(self.options['rho'])
        stemam = Constant(self.options['use_temam'])

        zero = Constant((0.,)*self.ndim)

        z = TestFunction(self.W)
        (v, q) = split(z)
        (u, p) = split(self.w)

        a = inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u, v)*dx - \
            p*div(v)*dx + q*div(u)*dx
        a += stemam*0.5*rho*div(u)*dot(u, v)*dx
        L = dot(zero, v)*dx

        # temporary! dirty hack!
        n = FacetNormal(self.mesh)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)
        self.energy_form = (inner(mu*grad(u), grad(u))*dx +
                            rho*dot(grad(u)*u, u)*dx +
                            stemam*0.5*rho*div(u)*dot(u, u)*dx -
                            0.5*rho*0.5*(dot(u, n) - abs(dot(u, n)))*dot(u, u)*ds(2)
                            )

        a_bc, L_bc = self._form_weak_bcs(self.w)
        a_bfs = self._backflowstab_nonlin()

        a = sum([a] + a_bc + a_bfs)
        L = sum([L] + L_bc)

        F = a - L
        J = derivative(F, self.w)

        self.nls_form = [F, J]

        return self

    def bilinear_form(self):
        ''' Build residual F and residual Jacobian J of the *linearized*
        variational problem, by means of the Picard and the Newton method.

        Attributes:
            ls_form         tuple (F, J), Picard linearization
            qnls_form       tuple (F, J), Newton linearization
        '''
        mu = Constant(self.options['mu'])
        rho = Constant(self.options['rho'])
        stemam = Constant(self.options['use_temam'])
        # sbf = Constant(self.options['use_backflowstab'])

        zero = Constant((0.,)*self.ndim)
        # n = FacetNormal(self.mesh)

        # ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        w = TrialFunction(self.W)
        (u, p) = split(w)
        (v, q) = TestFunctions(self.W)

        (u0, p0) = split(self.w)

        a = (inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u0, v)*dx -
             p*div(v)*dx + q*div(u)*dx)
        a += stemam*0.5*rho*div(u0)*dot(u, v)*dx

        L = dot(zero, v)*dx

        # Newton extra term for convection and Temam stabilization
        nc = (rho*dot(grad(u0)*u, v)*dx + stemam*0.5*rho*div(u)*dot(u0, v)*dx)

        a_bc, L_bc = self._form_weak_bcs(w)
        a_bfs, a_bfs_nl = self._backflowstab_lin()

        a = sum([a] + a_bc + a_bfs)
        L = sum([L] + L_bc)

        F = action(a - L, self.w)
        J = a
        Jnc = sum([J + nc] + a_bfs_nl)
        self.ls_form = [F, J]
        self.qnls_form = [F, Jnc]

        return self

    def boundary_conditions(self):
        ''' Process boundary conditions.
        Boundary conditions are defined by means of a list of dictionaries of
        the form
            [id1, { settings... },
             id2: { settings... },
                ...],
        where id1 and id2 are boundary indicators, matching self.bnds.
        **NOTE**: id=0 is reserved for INTERIOR EDGES. All boundaries need
        indicators id > 0!

        The settings of each boundary condition is a dictionary with keys

            'preset':  (Optionally) selects a predefined boundary condition.
                Each preset requires corresponding parameters set via the key
                'value': Possible options:

                    'noslip': No-slip BC with
                        'value' = float/Expression, or tuple with len == ndim
                    'driven_lid': regularized driven lid,
                        'value' = U
                    'inflow': parabolic inflow BC, requires
                        'value' = {'R': radius(optional), 'U': u_max,
                                    'symmetric': boolean}
                            Set 'symmetric' = True for half-parabola with u_max
                            at bottom, and False for full parabolic profile
                    'outflow': stress outlet with
                        'value' = pn
                    'symmetry': symmetry boundary condition

            'method': Method of imposing Dirichlet BCs. Nitsche's weak method
                is applied wrt normal and tangential compontents of the
                velocity vector, whereas standard essential BCs by constraining
                the function space (default) are set wrt to (x, y, z)
                components. If a single value is found as the 'value' item, it
                is applied on all components.
                options:
                    'nitsche':  requires 'value' = (normal, tangential)
                    'essential': requires 'value' = (ux, uy, uz)
                        or 'value' = u_all.
                    Compatible with 'preset'.

            'type': Set type of boundary condition. Self-explaining items:
                'dirichlet', 'neumann', 'navierslip'

            'value': The boundary value to be applied by the method
                specified above. Can be a single value or tuple (n, t) or (x,
                y, (z)) of numbers, DOLFIN Constants or Expressions.
                If combined with 'preset', value needs to be passed
                accordingly.
                Neumann accepts (n, t)
                Dirichlet essential (x, y, (z))
                Dirichlet Nitsche (n, )   # TODO extend for cartesian coords?
                                           # e.g. for 3D driven cavity lid
                                           # TODO: add flag 'cartesian'
                Navier-Slip: tba

        Attributes:
            bcs:        list of DirichletBC objects
            bcs_weak    list of weak Neumann, Navier-Slip or Nitsche BCs

        '''
        self.bcs = []
        self.bcs_neumann = []
        self.bcs_nitsche = []
        self.bcs_navierslip = []
        self.bcs_transpiration = []
        bc_lst = copy.deepcopy(self.options['boundary_conditions'])
        #  bc_lst = self.options['boundary_conditions']
        self.bc_lst = bc_lst

        self.check_boundary_conditions()

        for bc in bc_lst:
            if 'preset' in bc and bc['preset']:
                self._preset_bc_selector(bc)
            elif (bc['type'] == 'dirichlet' and bc['method'] == 'essential'):
                self._proc_dirichlet_bc(bc)
            elif (bc['type'] == 'dirichlet' and bc['method'] == 'nitsche'):
                self._proc_nitsche_bc(bc)
            elif bc['type'] == 'neumann':
                self._proc_neumann_bc(bc)
            elif bc['type'] == 'navierslip':
                raise Exception('This should never happen! Specify navier-slip'
                                ' BCs via preset interface')
                self._proc_navierslip_bc(bc)

        if self.options['fix_pressure']:
            self._proc_pressure_point_bc()

        return self

    def _form_weak_bcs(self, w):
        ''' Add weak boundary conditions (Neumann, Nitsche, Navier-Slip) to
        LHS & RHS of variational forms.
        'w' is passed (TrialFunction for bilinear_, Function self.w for
        nonlinear_) so that the same functions can be used for linearized and
        nonlinear solvers.

        Args:
            w   Function (nonlinear), TrialFunction (bilinear)

        Return:
            a, L    lists of BC contributions to a, L
        '''
        alist_neu, Llist_neu = self._create_neumann_form(w)
        alist_nit, Llist_nit = self._create_nitsche_form(w)
        alist_nav = self._create_navierslip_form(w)
        alist_nav += self._create_transpiration_form(w)

        return alist_neu + alist_nit + alist_nav, Llist_neu + Llist_nit

    def _create_neumann_form(self, w):
        ''' Create contributions to (a, L) due to Neumann BCs.

        Args:
            w       Function (nonlin) or TrialFunction (lin)

        Return:
            a       list of bilinear form terms (natural boundary integral)
            L       list of RHS contributions
        '''
        (v, q) = TestFunctions(self.W)
        (u, p) = split(w)

        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        n = FacetNormal(self.mesh)
        a = []
        L = []

        for bc in self.bcs_neumann:
            bid = bc[0]
            val = bc[1]
            if val == [None, 0.]:
                # this was the case for symmetry bcs, should not happen
                # anymore
                warnings.warn('NORMAL NEUMANN BC CONFLICTING WITH NITSCHE BC')
                assert True, 'redundant case. avoid. nothing done.'
                # a.append(-dot(mu*grad(u)*n - p*n, n)*dot(v, n)*ds(bid))
            elif ((self.is_Constant(val) or self.is_Expression(val)) and not
                  val.ufl_shape):   # scalar Expression or Constant
                L.append(val*dot(v, n)*ds(bid))
            else:
                raise Exception('Neumann BC data invalid!')

        return a, L

    def _create_nitsche_form(self, w):
        ''' Create contributions to (a, L) due to weak Dirichlet BCs via the
        Nitsche method.
        According to the options, the positivity/stability terms are
        skew-symmetric, e.g.,
            -<u-g, mu*grad(v)*n + q*n>
        (pressure cancels out when (v,q) = (u,p)) or "positivity" ensuring
        (TODO: check this!), e.g.,
            +<u-g, mu*grad(v)*n - q*n>,
        (full integral cancels out for (v,q) = (u,p)).

        Attributes:
            self.bcs_nitsche    list of shape [bid, bcval] where bcval is a
                                dolfin Constant or Expression of len 1 (normal
                                component imposed) or ndim (full cartesian
                                components)

        Args:
            w       Function (nonlin) or TrialFunction (lin)

        Return:
            a       list of bilinear form terms (natural boundary integral)
            L       list of RHS contributions
        '''
        a = []
        L = []

        if not self.bcs_nitsche:
            return a, L

        (u, p) = split(w)
        (v, q) = TestFunctions(self.W)
        n = FacetNormal(self.mesh)
        h = CellSize(self.mesh)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        beta1 = Constant(self.options['nitsche']['beta1'])
        beta2 = Constant(self.options['nitsche']['beta2'])
        met = self.options['nitsche']['method']
        mu = Constant(self.options['mu'])

        for bc in self.bcs_nitsche:
            bid = bc[0]
            val = bc[1]
            if not val.ufl_shape:
                # normal component only
                a.append((
                    -dot(mu*grad(u)*n - p*n, n)*dot(v, n) +
                    beta1/h*mu*dot(u, n)*dot(v, n) +
                    beta2/h*dot(u, n)*dot(v, n)
                )*ds(bid))
                L.append(
                    beta1/h*mu*val*dot(v, n)*ds(bid) +
                    beta2/h*val*dot(v, n)*ds(bid)
                )
                if met == 0:
                    a.append(-dot(mu*grad(v)*n + q*n, n)*dot(u, n)*ds(bid))
                    L.append(-dot(mu*grad(v)*n + q*n, n)*val*ds(bid))
                elif met == 1:
                    a.append(+dot(mu*grad(v)*n - q*n, n)*dot(u, n)*ds(bid))
                    L.append(+dot(mu*grad(v)*n - q*n, n)*val*ds(bid))

            elif len(val) == self.ndim:
                a.append((
                    -dot(mu*grad(u)*n - p*n, v) +
                    beta1/h*mu*dot(u, v) +
                    beta2/h*dot(u, n)*dot(v, n)
                    # beta2/h*dot(u, v)
                )*ds(bid))
                L.append(
                    beta1/h*mu*dot(val, v)*ds(bid) +
                    beta2/h*dot(val, v)*ds(bid)
                )
                if met == 0:
                    a.append(-dot(mu*grad(v)*n + q*n, u)*ds(bid))
                    L.append(-dot(mu*grad(v)*n + q*n, val)*ds(bid))
                elif met == 1:
                    a.append(+dot(mu*grad(v)*n - q*n, u)*ds(bid))
                    L.append(+dot(mu*grad(v)*n - q*n, val)*ds(bid))
            else:
                raise Exception('Invalid shape of BC value')

        return a, L

    def _create_navierslip_form(self, w):
        ''' Create contributions to (a, L) due to Navier-slip BCs.

        Attributes:
            self.bcs_navierslip     list of shape [bid, bcval] where bcval is a
                                    scalar dolfin Constant or Expression

        Args:
            w       Function (nonlin) or TrialFunction (lin)

        Return:
            a       list of bilinear form terms (natural boundary integral)
        '''
        a = []

        if not self.bcs_navierslip:
            return a

        (u, p) = split(w)
        (v, q) = TestFunctions(self.W)
        n = FacetNormal(self.mesh)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        for bc in self.bcs_navierslip:
            bid = bc[0]
            val = bc[1]
            assert self.is_Constant(val) or self.is_Expression(val), (
                'Gamma (val) expected to be of type Constant or Expression')

            a.append(-val*(dot(u, v) - dot(u, n)*dot(v, n))*ds(bid))

        return a

    def _create_transpiration_form(self, w):
        ''' Create contributions to (a, L) due to transpiration BCs.

        Attributes:
            self.bcs_transpiration  list of shape [bid, bcval] where bcval is a
                                    scalar dolfin Constant or Expression

        Args:
            w       Function (nonlin) or TrialFunction (lin)

        Return:
            a       list of bilinear form terms (natural boundary integral)
        '''
        a = []

        if not self.bcs_transpiration:
            return a

        (u, p) = split(w)
        (v, q) = TestFunctions(self.W)
        n = FacetNormal(self.mesh)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        for bc in self.bcs_transpiration:
            bid = bc[0]
            val = bc[1]
            assert self.is_Constant(val) or self.is_Expression(val), \
                'Gamma (val) expected to be of type Constant or Expression'

            a.append(val*dot(u, n)*dot(v, n)*ds(bid))

        return a

    def _backflowstab_lin(self):
        ''' Build backflow stabilization terms for outlet and for weak
        Dirichlet BCs.

        Args:

        Return:
            alist       contribution to variational form (Picard terms)
            alist_newt  Newton extra terms
        '''
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)
        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)
        (u0, p0) = split(self.w)

        n = FacetNormal(self.mesh)
        rho = self.options['rho']
        alist = []
        alist_newt = []
        if self.options['backflowstab']['outlet']:
            if self._bid_outlet:
                bid = self._bid_outlet
                alist.append(-0.5*rho*0.5 *
                             (dot(u0, n) - abs(dot(u0, n)))*dot(u, v)*ds(bid)
                             )
                # -0.5*rho*self.abs_n(dot(u0, n))*dot(u, v)*ds(bid)
                alist_newt.append(
                    -0.5*rho*0.5*(
                        # dot(u, n)*dot(u, v) - abs(dot(u, n))*dot(u, v)
                        dot(u, n)*dot(u0, v) -
                        sign(dot(u0, n))*dot(u, n)*dot(u0, v)
                        # + dot(u0, n)*dot(u, v)  # included in Picard term
                        # -abs(dot(u0, n))*dot(u, v) # included in Picard term
                        # last line equals
                        # sign(dot(u0, n))*dot(u0, n)*dot(u, v)
                    )*ds(bid)
                )
                # print('outlet stab: bid {0}'.format(bid))

        # XXX: hacked?? loop through nitsche AND TRANSPIRATION
        for bc in self.bcs_nitsche + self.bcs_transpiration:
            assert not bc[0] == self._bid_outlet, (
                'Outlet ID == Nitsche ID. This should never happen.')

            bid = bc[0]
            if self.options['backflowstab']['nitsche'] == 0:
                # no backflow stabilization on Nitsche boundaries
                pass
            elif self.options['backflowstab']['nitsche'] == 1:
                alist.append(
                    -0.5*rho*dot(u0, n)*dot(u, v)*ds(bid)
                )
                alist_newt.append(
                    -0.5*rho*dot(u, n)*dot(u0, v)*ds(bid)
                )

            elif self.options['backflowstab']['nitsche'] == 2:
                alist.append(
                    -0.5*rho*0.5 *
                    (dot(u0, n) - abs(dot(u0, n)))*dot(u, v)*ds(bid)
                )
                alist_newt.append(
                    -0.5*rho*0.5*(
                        dot(u, n)*dot(u0, v) -
                        sign(dot(u0, n))*dot(u, n)*dot(u0, v)
                    )*ds(bid)
                )

        return alist, alist_newt

    def _backflowstab_nonlin(self):
        ''' Build backflow stabilization terms for outlet and for weak
        Dirichlet BCs.

        Return:
            alist   list of contributions to variational form
        '''
        (v, q) = TestFunctions(self.W)
        (u, p) = split(self.w)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        n = FacetNormal(self.mesh)
        # h = CellSize(self.mesh)
        rho = self.options['rho']
        alist = []
        if self.options['backflowstab']['outlet']:
            if self._bid_outlet:
                bid = self._bid_outlet
                alist.append(
                    -0.5*rho*self.abs_n(dot(u, n))*dot(u, v)*ds(bid)
                )

        # XXX: hacked?? loop through nitsche AND TRANSPIRATION
        for bc in self.bcs_nitsche + self.bcs_transpiration:
            assert not bc[0] == self._bid_outlet, (
                'Outlet ID == Nitsche ID. This should never happen.')
            bid = bc[0]

            if self.options['backflowstab']['nitsche'] == 0:
                # no backflow stabilization at Nitsche boundaries
                pass
            elif self.options['backflowstab']['nitsche'] == 1:
                alist.append(
                    -0.5*rho*dot(u, n)*dot(u, v)*ds(bid)
                )
            elif self.options['backflowstab']['nitsche'] == 2:
                alist.append(
                    -0.5*rho*self.abs_n(dot(u, n))*dot(u, v)*ds(bid)
                )

        return alist

    def _preset_bc_selector(self, bc):
        ''' Prepare preset boundary condition.
        Create data sets and call _proc_dirichlet_bc and/or
        _proc_neumann_bc as necessary.

        Recognized options:
            noslip
            inlet
            driven_lid
            outlet
            symmetry
            navierslip
            navierslip_transpiration

        Args:
            bc     list of BCs {'id': bid, 'method', 'value',...}
        '''
        preset = bc['preset']

        if preset == 'inlet':
            # get direction of boundary, assuming it is (more or less) straight
            self._preset_inlet_bc(bc)

        elif preset == 'driven_lid':
            self._preset_driven_lid_bc(bc)

        elif preset == 'noslip':
            self._preset_noslip_bc(bc)

        elif preset == 'outlet':
            self._preset_outlet_bc(bc)

        elif preset == 'symmetry':
            self._preset_symmetry_bc(bc)

        elif preset == 'navierslip':
            self._preset_navierslip_bc(bc)
            self._preset_no_penetration_bc(bc)

        elif preset == 'navierslip_transpiration':
            self._preset_navierslip_bc(bc)
            self._preset_transpiration_bc(bc)

        pass

    def _preset_inlet_bc(self, bc):
        ''' Process preset inlet boundary condition, create format that
        function _proc_dirichlet_bc understands.

        Args:
            bc      dictionary describing one boundary condition
        '''
        bid = bc['id']
        if 'method' in bc and bc['method'] == 'nitsche':
            warnings.warn('Inlet BCs are imposed strongly. Ignoring '
                          'Nitsche setting.')

        if not self.is_Expression(bc['value']):
            # shallow copy for ReynoldsContinuation
            symmetric = bc['symmetric'] if 'symmetric' in bc else False
            bnd_dir = self._get_boundary_orientation(bid)
            x0, r0 = self._get_inlet_parabola_coef(bid, bnd_dir, symmetric)
            r0 = bc['value']['R'] if 'R' in bc['value'] else r0

            # ind: list of indices != bnd_dir(inflow direction)
            ind = range(self.ndim)
            ind.remove(bnd_dir)
            if self.ndim == 2:
                assert bnd_dir in (0, 1), ('bnd_dir not in (0, 1)!')
                inflow_str = 'U*(1 - pow(x[{0}] - x0, 2)/(R*R))'.format(ind[0])
            elif self.ndim == 3:
                warnings.warn('3D paraboloidal inlet profile only valid for '
                              'circular cross sections')
                inflow_str = ('U/(R*R)*(R*R - pow(x[{0}] - x0, 2) -'
                              'pow(x[{1}] - x0, 2))'.format(ind[0], ind[1]))

            inflow_lst = ['0.0']*self.ndim
            inflow_lst[bnd_dir] = inflow_str
            inflow = Expression(inflow_lst, U=bc['value']['U'], x0=x0, R=r0,
                                degree=2)
            bc['value'] = inflow

        if 'method' in bc and bc['method'] == 'nitsche':
            assert True, 'Inlet only possible via strong Dirichlet BCs.'
            # self._proc_nitsche_bc(bc)
        else:
            self._proc_dirichlet_bc(bc)
        pass

    def _preset_driven_lid_bc(self, bc):
        ''' Process preset driven lid boundary condition, create format that
        function _proc_dirichlet_bc understands.
        Define a 2D or 3D Expression for a *regularized* velocity
            u = U/R^4*(R^4 - (x - x0)^4), v = 0
        or (assuming cube),
            u = U/R^4*(R^4 - (x - x0)^4 - (z - x0)^4, v = 0, w = 0
        where x0 is the center point of the boundary and R its straight line
        distance to the bounds of the surface or line.

        Args:
            bc      dictionary describing one boundary condition
        '''
        bid = bc['id']
        bnd_dir = self._get_boundary_orientation(bid)
        x0, r0 = self._get_inlet_parabola_coef(bid, bnd_dir)
        warnings.warn(('Driven lid assumed to be top boundary, moving in '
                      'the x direction'))

        assert bnd_dir == 1, 'Lid normal != e_y. Apply lid to top bnd.'

        if self.ndim == 2:
            lid_str = 'U/R4*(R4 - pow(x[0] - x0, 4))'
        elif self.ndim == 3:
            lid_str = ('U/R4*(R4 - pow(x[0] - x0, 4) - pow(x[2] - x0, 4))')
            warnings.warn('check 3D driven lid expression. UNTESTED.')

        lid_lst = ['0.0']*self.ndim
        lid_lst[0] = lid_str

        lid = Expression(lid_lst, U=bc['value']['U'], x0=x0, R4=r0**4,
                         degree=4)
        bc['value'] = lid

        if 'method' in bc and bc['method'] == 'nitsche':
            self._proc_nitsche_bc(bc)
        else:
            self._proc_dirichlet_bc(bc)
        pass

    def _preset_noslip_bc(self, bc):
        ''' Process preset no-slip boundary condition, create format that
        function _proc_dirichlet_bc understands.

        Args:
            bc      dictionary describing one boundary condition
        '''
        if 'method' in bc and bc['method'] == 'nitsche':
            self._proc_nitsche_bc(bc)
        else:
            self._proc_dirichlet_bc(bc)

        pass

    def _preset_outlet_bc(self, bc):
        ''' Process preset pressure outlet boundary condition, create format
        that function _proc_neumann_bc understands. (Just call function)

        Args:
            bc      dictionary describing one boundary condition
        '''
        self._proc_neumann_bc(bc)
        self._bid_outlet = bc['id']
        pass

    def _preset_symmetry_bc(self, bc):
        ''' Process preset symmetry boundary condition, create format that
        functions _proc_dirichlet_bc and _proc_neumann_bc understand.

        Create Dirichlet BC for u_n = 0 and Neumann for n.sigma.t = 0.

        Args:
            bc      dictionary describing one boundary condition
        '''
        bid = bc['id']
        # bc_n = {'id': bc['id'],
        #         'type': 'neumann',
        #         'value': [None, 0.]
        #         }
        bc_d = {'id': bc['id'],
                'method': bc['method'] if 'method' in bc else 'essential'
                }
        if bc_d['method'] == 'nitsche':
            bc_d['value'] = [0., None]
            self._proc_nitsche_bc(bc_d)
        else:
            bnd_dir = self._get_boundary_orientation(bid)
            # bnd_dir is normal direction
            val = [None]*self.ndim
            val[bnd_dir] = 0.
            bc_d['value'] = val
            self._proc_dirichlet_bc(bc_d)

        # XXX do nothing: normal component is taken care of by the Dirichlet BC
        #       and tangential component is zero.
        # self._proc_neumann_bc(bc_n)
        pass

    def _preset_navierslip_bc(self, bc):
        ''' Process preset Navier-slip boundary condition, create format that
        function _proc_navierslip_bc understands.

        Args:
            bc      dictionary describing one boundary condition
        '''

        bid = bc['id']
        gm = bc['value']['gm']
        R_inn = bc['value']['R'] if 'R' in bc['value'] else 0
        #  R_out = R_inn + bc['value']['dR']
        dR = bc['value']['dR']

        # gm_pois = 2.0*self.options['mu']*R_inn/(R_inn**2 - R_out**2)
        # # gm_const = Constant(gm_pois*gm)
        # gm_expr = Expression('G0*a', G0=gm_pois, a=gm)

        # XXX define gm_expr as Expression of variables: mu, Ri, Ro, a=factor
        # Now gm_expr has a starting value, but each variable can be modified
        # separately.
        #  gm_expr = Expression('2.0*a*mu*R_i/(R_i*R_i - R_o*R_o)',
        #                    a=gm, mu=self.options['mu'], R_i=R_inn, R_o=R_out)
        # XXX 22/09/16      write gamma in terms of R_i, dR only!
        if R_inn > 0:
            gm_expr = Expression('2.0*a*mu*R_i/(-2*R_i*dR - dR*dR)',
                                 a=gm, mu=self.options['mu'], R_i=R_inn, dR=dR,
                                 degree=1)
        else:
            gm_expr = Expression('2.0*a*mu*x[1]/(-2*x[1]*dR - dR*dR)',
                                 a=gm, mu=self.options['mu'], dR=dR, degree=1)

        self._proc_navierslip_bc([bid, gm_expr])
        pass

    def _preset_no_penetration_bc(self, bc):
        ''' Process no-penetration boundary condition.

        Args:
            bc      dictionary describing one boundary condition
        '''

        bc_d = {'id': bc['id'],
                'method': bc['method'] if 'method' in bc else 'nitsche'
                }
        if bc_d['method'] == 'nitsche':
            bc_d['value'] = [0., None]
            self._proc_nitsche_bc(bc_d)
        else:
            warnings.warn('Using strong DBC for non-penetration BC')
            bnd_dir = self._get_boundary_orientation(bc['id'])
            # bnd_dir is normal direction
            val = [None]*self.ndim
            val[bnd_dir] = 0.
            bc_d['value'] = val
            self._proc_dirichlet_bc(bc_d)

        pass

    def _preset_transpiration_bc(self, bc):
        ''' Process preset transpiration boundary condition, create format that
        function _proc_transpiration_bc understands.

        Args:
            bc      dictionary describing one boundary condition
        '''
        bid = bc['id']
        resistance = bc['value']['beta']

        #  res_expr = Expression('1.0*val', val=resistance)
        res = Constant(resistance)

        self._proc_transpiration_bc([bid, res])

        pass

    def _get_boundary_orientation(self, bid):
        ''' Get boundary orientation from boundary  MeshFunction 'bnds'.
        Extracts coordinates of boundary points and and finds constant
        cartesian coordinate direction, if any. This direction equals the
        normal direction. Limited to horizontal and vertical boundaries.

        Args:
            bid     boundary indicator

        Returns:
            imin    component index with minimum coordinate standard deviation

        Example:
            _get_boundary_orientation(1) == 0 means, that along boundary 1, the
            x (i=0) coordinate is constant; i.e. the normal vector is n = e_x.
            A return value imin = 1 means that y = const, the boundary is
            parallel to the X-Z plane.
        '''
        # http://fenicsproject.org/qa/9135/obtain-coordinates-defined-by-mesh-function?show=9135#q9135

        It_facet = SubsetIterator(self.bnds, bid)
        pts = []
        for c in It_facet:
            pts.append([c.midpoint().x(), c.midpoint().y(), c.midpoint().z()])
        pts_std = np.array(pts).std(axis=0)
        if self.ndim == 2:
            pts_std = pts_std[:self.ndim]
        imin = np.argmin(pts_std)
        assert pts_std[imin] <= 10*DOLFIN_EPS, (
            'Found minimal std = {0} in coordinate direction i = {1}, does '
            'not seem to be a straight line'.format(pts_std[imin], imin))

        return imin

    def _get_inlet_parabola_coef(self, bid, bnd_dir, symmetric=False):
        ''' Get radius and center point for parabolic inflow profile.
        3D: assume circular inlet!
        Symmetric: always assume symmetry axis is (0, 0) center line
        Also require that R > 0!

        Args:
            bid (int)       boundary indicator
            bnd_dir (int)   normal direction (x,y,z) = (0,1,2) at boundary

        Returns:
            R               Radius w.r.t. midpoint
            x0              center/mid point
        '''

        It_facet = SubsetIterator(self.bnds, bid)
        pts = []
        for c in It_facet:
            for v in vertices(c):
                ptsi = [v.point().x(), v.point().y(), v.point().z()]
                ptsi.pop(bnd_dir)
                pts.append(ptsi)

        pts = np.array(pts)
        assert pts.shape[1] == 2, (
            'Something wrong with dimensions. Index bnd_dir not deleted?')

        pts1 = pts[:, 0]
        if symmetric:
            x0 = pts1.min()
            assert x0 == 0.0, 'Symmetric: x0 expected to be 0.0'
            R = pts1.max()
        else:
            x0 = 0.5*(pts1.max() + pts1.min())
            R = 0.5*(pts1.max() - pts1.min())

        if self.ndim == 3 and symmetric:
            assert pts[:, 0].max() - pts[:, 1].max() <= DOLFIN_EPS, (
                'Symmetric 3D section does not seem circular AND with'
                ' *positive radius*')

        return x0, R

    def _proc_navierslip_bc(self, bc):
        ''' Prepare Navier-Slip Gamma boundary condition '''
        bid = bc[0]
        bcval = bc[1]

        if self.bcs_navierslip is None:
            self.bcs_navierslip = []

        if type(bcval) in (int, float):
            bcval = Constant(bcval)
        elif not (self.is_Constant(bcval) or self.is_Expression(bcval)):
            raise Exception('Navier-Slip friction coefficient must be a number'
                            ' or a dolfin Constant/Expression.')

        self.bcs_navierslip.append([bid, bcval])
        pass

    def _proc_transpiration_bc(self, bc):
        ''' Prepare transpiration boundary condition '''
        bid = bc[0]
        bcval = bc[1]

        if self.bcs_transpiration is None:
            self.bcs_transpiration = []

        if type(bcval) in (int, float):
            bcval = Constant(bcval)
        elif not (self.is_Constant(bcval) or self.is_Expression(bcval)):
            raise Exception('Transpiration resistance coefficient must be a '
                            'number or a dolfin Constant/Expression.')

        self.bcs_transpiration.append([bid, bcval])
        pass

    def _proc_dirichlet_bc(self, bc):
        ''' Create Dirichlet boundary condition and appends to instance
        self.bcs list.
        Check if bc value is given as
            1. list with len == ndim
                - of int/float
                - containing Nones AND int/float/Expression/Constant
        or  2. Expression/Constant with len == ndim
        and treat appropriately.

        Args:
            bc          BC dict, with keys {'id', 'method', 'type', 'value'}

        Attributes:
            self.bcs    appends BC to instance self.bcs list
        '''

        bid = bc['id']
        bcval = bc['value']
        if type(bcval) is tuple:
            bcval = [b for b in bcval]

        assert len(bcval) == self.ndim, (
            'BC definition does not match geometric dimensions. Use '
            '(val, None, None) if no BC should be set.')
        # Values given as numbers or list of numbers => convert
        if type(bcval) is list:
            if all(isinstance(b, (int, float)) for b in bcval):
                bcval = Constant(bcval)
                V = self.W.sub(0)
                if self.is_enriched(V):
                    bcval = project(bcval, V.collapse(), solver_type='lu')

                self.bcs.append(DirichletBC(V, bcval, self.bnds, bid))

            elif None in bcval:
                # get indices where not None
                inone = [i for i, x in enumerate(bcval) if x is not None]
                for i in inone:
                    if isinstance(bcval[i], (int, float)):
                        bcval[i] = Constant(bcval[i])
                    if (self.is_Constant(bcval[i]) or
                            self.is_Expression(bcval[i])):
                        Vi = self.W.sub(0).sub(i)
                        if self.is_enriched(Vi):
                            bcval[i] = project(bcval[i], Vi.collapse(),
                                               solver_type='lu')
                        self.bcs.append(DirichletBC(Vi, bcval[i], self.bnds,
                                                    bid))
            else:
                raise Exception('Type in BC value array not recognized. '
                                'Maybe mixed numbers with DOLFIN Const/Expr?')
        else:
            # not a list
            if self.is_Expression(bcval) or self.is_Constant(bcval):
                if bcval.ufl_shape and len(bcval) == self.ndim:
                    V = self.W.sub(0)
                    if self.is_enriched(V):
                        bcval = project(bcval, V.collapse(), solver_type='lu')

                    self.bcs.append(DirichletBC(V, bcval, self.bnds, bid))

                else:
                    raise Exception('len(bcval) == ndim required!')
            else:
                raise Exception('bcval was expected to be dolfin Expression '
                                'or Constant')
        pass

    def _proc_nitsche_bc(self, bc):
        ''' Prepare Nitsche Dirichlet boundary condition.
        Assume that BC value is given as a list or Constant/Expression of the
        cartesian vector components [u1, u2, (u3)].
        The normal component can be specified via [g, None] (None referring to
        the tangential component FIXME).

        Possible values:
            set full velocity vector
                [u1, u2, (u3)]
                Expression((u1, u2, u3)
                Constant((u1, u2, u3)
            only normal component
                [val1, None]

        Args:
            bc          BC dict, with keys {'id', 'method', 'value'}
        '''
        # FIXME: MAKE CLEAR NORMAL/TANGENT INDICATOR!
        if self.bcs_nitsche is None:
            self.bcs_nitsche = []

        bid = bc['id']

        bcval = bc['value']
        if type(bcval) is tuple:
            bcval = [bc for bc in bcval]

        if self.is_Constant(bcval) or self.is_Expression(bcval):
            if bcval.ufl_shape and len(bcval) == self.ndim:
                self.bcs_nitsche.append([bid, bcval])
            else:
                raise Exception('Nitsche BC of type Const/Expr expected to '
                                'have len=ndim')

        elif type(bcval) is list:
            if (all(type(b) in (int, float) for b in bcval) and len(bcval) ==
                    self.ndim):
                self.bcs_nitsche.append([bid, Constant(bcval)])
            elif bcval[1] is None and type(bcval[0]) in (int, float):
                self.bcs_nitsche.append([bid, Constant(bcval[0])])
            else:
                raise Exception('Nitsche BCs of list type need to be given in '
                                'form [num1, num2, (num3)] or [num1, None].')
        else:
            raise Exception('Type of bc value needs to be list or Const/Expr.')

        pass

    def _proc_pressure_point_bc(self):
        ''' Prepare pressure point Dirichlet boundary condition.
        Set zero automatically.
        '''
        pt = self.options['fix_pressure_point']
        if len(pt) == self.mesh.topology().dim():
            bc = DirichletBC(self.W.sub(1), 0.0, geom.Corner(pt),
                             method='pointwise')
            self.bcs.append(bc)
        else:
            raise Exception('Dimension of pressure BC point coordinates != '
                            'mesh dimension.')
        pass

    def _proc_neumann_bc(self, bc):
        ''' Prepare Neumann boundary condition.
            Assume a scalar value or a [normal, tangential]-list is given.

            Possible values/combinations:
                scalar p0:
                    'p0*dot(v, n)*ds(bid)'
                vector [tn, tt]:
                    'tn.n*dot(v, n)*ds + dot(tt, (v - v.nn))*ds'  (check?)
                    > tn == None, tt == 0   (symmetry)
                TODO: Other meaningful combinations ??

            Args:
                bc     boundary condition dict
        '''
        if self.bcs_neumann is None:
            self.bcs_neumann = []

        bid = bc['id']

        bcval = bc['value']
        if type(bcval) is tuple:
            bcval = [bc for bc in bcval]

        if type(bcval) in (int, float):
            bcval = Constant(bcval)
            self.bcs_neumann.append([bid, bcval])

        elif type(bcval) is list:
            if bcval == [None, 0]:
                warnings.warn('zero neumann tangential. why add 0? ommit!')
                self.bcs_neumann.append([bid, bcval])
            else:
                raise Exception('Neumann BC values must be given as scalar or '
                                '[normal, tangential] list. Only valid option '
                                'at the moment: [None, 0]. Slip-BCs must be '
                                'set via navierslip.')
        else:
            raise Exception('Type of bc value not recognized')

        # OLD CODE. Too complex and general for specific needs.
        # elif type(bcval) is list and len(bcval) == 2:
        #     if all(type(b) in (int, float) for b in bcval):
        #         bcval = Constant(bcval)
        #         self.bcs_neumann.append([bid, bcval])
        #     elif None in bcval:
        #         # get indices where not None
        #         # NOTE: knowing that len == 2, this can be done much simpler
        #         #   but possible extension to cartesian 3D formulation ...
        #         inone = [i for i, x in enumerate(bcval) if x is not None]
        #         for i in inone:
        #             if isinstance(bcval[i], (int, float)):
        #                 bcval[i] = Constant(bcval[i])
        #             if (self.is_Constant(bcval[i]) or
        #                     self.is_Expression(bcval[i])):
        #                 self.bcs_neumann.append([bid, i, bcval[i]])
        #             else:
        #                 raise Exception('Value type not recognized.')
        #     else:
        #         raise Exception('Type in BC value array not recognized. '
        #                         'Maybe mixed numbers with Const/Expr?')
        # else:
        #     raise Exception('Expected list of length 2,
        # [normal, tangential]')

        # assert len(self.bcs_neumann) == count + 1, (
        #     'No or more than 1 Neumann BCs set. Should be exactly 1')

        pass

    def check_boundary_conditions(self):
        ''' Check consistency of boundary conditions. '''
        if not self.W:
            raise Exception('Function space needs to be created prior to'
                            ' creating boundary conditions.')
        if not self.bnds:
            raise Exception('Boundary indicator MeshFunction not set!')

        bcs = self.options['boundary_conditions']
        bc_id = [bc['id'] for bc in bcs]

        indicators = np.unique(self.bnds.array())
        indicators = indicators[indicators > 0]
        if not np.unique(bc_id).sort() == indicators.sort():
            raise Exception('Mesh boundary indicators do not match boundary '
                            ' conditions IDs.')

        return self

    def is_enriched(self, V):
        ''' Check if the given (sub) function space has enriched elements. '''
        while V.num_sub_spaces():
            V = V.sub(0)
        check = (type(V.ufl_element()) ==
                 ufl.finiteelement.enrichedelement.EnrichedElement)
        return check

    def is_Expression(self, obj):
        ''' Check if object has type dolfin Expression '''
        return bool('Expression' in str(type(obj)))

    def is_Constant(self, obj):
        ''' Check if object has type dolfin Constant '''
        return type(obj) is dolfin.functions.constant.Constant

    def abs_n(self, x):
        return 0.5*(x - abs(x))
