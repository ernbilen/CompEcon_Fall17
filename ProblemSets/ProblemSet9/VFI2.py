import numba
import numpy as np
@numba.jit
def VFI_loop(V, e, betafirm, sizez, sizek, Vmat, pi):
    '''
    This function performs VFI loop using the numba method
    '''
    V_prime = np.dot(pi, V)
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            for k in range(sizek): # loop over k'
                Vmat[i, j, k] = e[i, j, k] + betafirm * V_prime[i, k]
    return Vmat


def main(theta, params, z_grid, kgrid, sizek, sizez, pi, a_k, w, psi):
    
    
    betafirm, delta, mu, sizez, a_l = params
    a_k, psi, rho, sigma, tao = theta
    # operating profits, op
    sizez = len(z_grid)
    op = np.zeros((sizez, sizek))
    for i in range(sizez):
        for j in range(sizek):
            op[i,j] = ((1 - a_l) * ((a_l / w) ** (a_l / (1 - a_l))) *
          ((kgrid[j] ** a_k) ** (1 / (1 - a_l))) * (z_grid[i] ** (1/(1 - a_l))))

    # firm cash flow, e    
    e = np.zeros((sizez, sizek, sizek))
    for i in range(sizez):
        for j in range(sizek):
            for k in range(sizek):
                e[i, j, k] = (op[i,j] - tao - kgrid[k] + ((1 - delta) * kgrid[j]) -
                           ((psi / 2) * ((kgrid[k] - ((1 - delta) * kgrid[j])) ** 2)
                            / kgrid[j]))

    # value function iteration
    VFtol = 1e-6
    VFdist = 7.0
    VFmaxiter = 3000
    V = np.zeros((sizez, sizek)) 
    Vmat = np.zeros((sizez, sizek, sizek))  
    Vstore = np.zeros((sizez, sizek, VFmaxiter)) 
    VFiter = 1


    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V    
        Vmat = VFI_loop(V, e, betafirm, sizez, sizek, Vmat, pi)
        Vstore[:, :, VFiter] = V.reshape(sizez, sizek,) 
        V = Vmat.max(axis=2) 
        PF = np.argmax(Vmat, axis=2) 
        Vstore[:,:, i] = V  
        VFdist = (np.absolute(V - TV)).max()  
        VFiter += 1
        
    VF = V

    # optimal capital stock k'
    optK = kgrid[PF]     
    
    return V, PF