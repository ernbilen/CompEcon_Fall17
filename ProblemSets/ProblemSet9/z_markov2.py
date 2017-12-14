# Simulate the Markov process - will make this a function so can call later
def sim_markov(z_grid, pi, num_draws):
    import numpy as np
    
    # draw some random numbers on [0, 1]
    np.random.seed(seed = 42)
    u = np.random.uniform(size=num_draws)

    # Do simulations
    z_discrete = np.empty(num_draws)  # this will be a vector of values
    # we land on in the discretized grid for z
    N = len(z_grid)
    oldind = int(np.ceil((N - 1) / 2))  # set initial value to median of grid
    z_discrete[0] = oldind
    for i in range(1, num_draws):
        sum_p = 0
        ind = 0
        while sum_p < u[i]:
            sum_p = sum_p + pi[ind, oldind]
#             print('inds =  ', ind, oldind)
            ind += 1
        if ind > 0:
            ind -= 1
        z_discrete[i] = ind
        oldind = ind

    return z_discrete