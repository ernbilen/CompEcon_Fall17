def main(sigma, mu, rho, sizez):

    '''
    Calculates and returns the grid of z values
    '''
    import numpy as np
    from scipy.stats import norm
    import scipy.integrate as integrate
    import numba
    import time



# calculate the grid for z
# compute cut-off values
    sigmaz = sigma / ((1 - rho ** 2) ** (1 / 2))
    N = 9  # number of grid points
    z_cutoffs = (sigmaz * norm.ppf(np.arange(N + 1) / N)) + mu
    z_grid = ((N * sigmaz * (norm.pdf((z_cutoffs[:-1] - mu) / sigmaz) -
              norm.pdf((z_cutoffs[1:] - mu) / sigmaz))) + mu)

# define function that we will integrate over
    def integrand(x, sigmaz, sigma, rho, mu, z_j, z_jp1):
        val = (np.exp((-1 * ((x - mu) ** 2)) / (2 * (sigmaz ** 2))) *
                   (norm.cdf((z_jp1 - (mu * (1 - rho)) - (rho * x)) / sigma) -
                    norm.cdf((z_j - (mu * (1 - rho)) - (rho * x)) / sigma)))
        return val

# compute transition probabilities
    pi = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            results = integrate.quad(integrand, z_cutoffs[i], z_cutoffs[i + 1],
                                     args=(sigmaz, sigma, rho, mu,
                                           z_cutoffs[j], z_cutoffs[j + 1]))
            pi[i, j] = (N / np.sqrt(2 * np.pi * sigmaz ** 2)) * results[0]
        
# transform the shocks that are in logs
    z_grid = [np.exp(x) for x in z_grid]


    return z_grid, pi