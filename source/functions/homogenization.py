import numpy as np
import sys
from dolfin import *

# Generate effective homogenized tensor for a domain with rank-2 laminations
def generate_efective_tensor(prms, subdomains, theta_c, theta_f, dofs, Deff, Deffp):

    e1x, e1y 	= prms['mesh']['lamination_direction_main']
    e2x, e2y 	= prms['mesh']['lamination_direction_cross']   
    ffx, ffy 	= prms['mesh']['main_fiber_direction']   
    ccx, ccy 	= prms['mesh']['cross_fiber_direction']
    ffx, ffy 	= prms['mesh']['main_fiber_direction']
    sigma_1     = prms['phys']['sigma_1']
    gamma       = prms['phys']['gamma']
    betha       = prms['phys']['betha']

    # TODO: deduce lamination direction form fiber and cross fiber direction
    e1 = np.matrix(str(e1x) + ';' + str(e1y))
    e2 = np.matrix(str(e2x) + ';' + str(e2y))
    ff = np.matrix(str(ffx) + ';' + str(ffy))
    cc = np.matrix(str(ccx) + ';' + str(ccy))

    sigma_2 = sigma_1/gamma
    sigma_c = pow(10, -betha)*sigma_1

    # healthy and collagen diffusion tensors and initialize effective tensors
    Dh   = sigma_1*ff*ff.transpose() + sigma_2*cc*cc.transpose()
    Dcol = np.matrix(str(sigma_c) + ' 0; 0 ' + str(sigma_c))

    # class used to assign different tensors to differents sub-domains
    class DiscontinuousTensor(Expression):# 	Class with methods to assign tensors to its respective materials
        def __init__(self, cell_function, tensors):
            self.cell_function = cell_function
            self.coeffs 	   = tensors
        def value_shape(self):
            return (2,2)
        def eval_cell(self, values, x, cell):
            subdomain_id = self.cell_function[cell.index]
            local_coeff  = self.coeffs[subdomain_id]
            local_coeff.eval_cell(values, x, cell)


    aux_1 = ((Dcol - Dh)*e1)*((Dcol - Dh).transpose()*e1).transpose()
    aux_2 = (Dcol*e1)[0]*e1[0] + (Dcol*e1)[1]*e1[1]
    aux_3 = (Dh*e1)[0]*e1[0]   + (Dh*e1)[1]*e1[1]

    tc_vec = theta_c.vector().array()
    tf_vec = theta_f.vector().array()


    kkk = 0
    print 'Generating First Effective Tensor Field: '
    for dof in dofs: 
        porcentaje = float(dof)/float((dofs.size))*100.0
        print '\r%.0f %%' % porcentaje,
        sys.stdout.flush()

        tensor_local = (1 - tc_vec[dof])*Dh + tc_vec[dof]*Dcol - tc_vec[dof]*(1 - tc_vec[dof])*aux_1/((1 - tc_vec[dof])*aux_2 + tc_vec[dof]*aux_3)

        Deffp.vector()[kkk + 0] = tensor_local[0,0]
        Deffp.vector()[kkk + 1] = tensor_local[0,1]
        Deffp.vector()[kkk + 2] = tensor_local[1,0]
        Deffp.vector()[kkk + 3] = tensor_local[1,1]
        kkk = kkk + 4
   
    print '\r100 % [COMPLETE]'
    
    aux_3 = (Dh*e2)[0]*e2[0]   + (Dh*e2)[1]*e2[1]
    Deffp_vec = Deffp.vector().array(); kkk = 0

   
    print 'Generating Final Effective Tensor Field: '
    for dof in dofs:        
        porcentaje = float(dof)/float((dofs.size))*100.0
        print '\r%.0f %%' % porcentaje,
        sys.stdout.flush()

        Deffp_local = np.matrix(str(Deffp_vec[kkk + 0]) + ' ' + str(Deffp_vec[kkk + 1]) + ';' + str(Deffp_vec[kkk + 2]) + ' ' + str(Deffp_vec[kkk + 3]))
        aux_1 = ((Deffp_local - Dh)*e2)*((Deffp_local - Dh).transpose()*e2).transpose()
        aux_2 = (Deffp_local*e2)[0]*e2[0] + (Deffp_local*e2)[1]*e2[1]

        tensor_local = (1 - tf_vec[dof])*Dh + tf_vec[dof]*Deffp_local - tf_vec[dof]*(1 - tf_vec[dof])*aux_1/((1 - tf_vec[dof])*aux_2 + tf_vec[dof]*aux_3)

        Deff.vector()[kkk + 0] = tensor_local[0,0]
        Deff.vector()[kkk + 1] = tensor_local[0,1]
        Deff.vector()[kkk + 2] = tensor_local[1,0]
        Deff.vector()[kkk + 3] = tensor_local[1,1]

        kkk = kkk + 4
    ndofs = 15032
    print '\r100 % [COMPLETE]'

    # diffusion tensors UFL versions
    Dh	    = Constant(((sigma_1,      0),	
                     (0,      sigma_2)))
    Dcol    = Constant(((sigma_c  , 0.0,),
                     (0.0,    sigma_c)))
    # Assign tensors where they belongs
    C = DiscontinuousTensor(subdomains, [Dh, Dcol])

    return Deff, C
