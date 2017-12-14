def main(params,w,a_k):

    '''
    This function calculates the grid space for capital
    '''
    import numpy as np
    from scipy.stats import norm
    import scipy.integrate as integrate
    import numba
    import time

    betafirm, delta, mu, sizez, a_l = params
    # calculate the grid for k
    dens = 5
    # put in bounds here for the capital stock space
    kstar = 11
    kbar = 2*kstar
    lb_k = 0.01
    ub_k = kbar
    krat = np.log(lb_k / ub_k)
    numb = np.ceil(krat / np.log(1 - delta))
    K = np.zeros(int(numb * dens))

    for j in range(int(numb * dens)):
        K[j] = ub_k * (1 - delta) ** (j / dens)
        kgrid = K[::-1]
        sizek = kgrid.shape[0]

    return kgrid, sizek