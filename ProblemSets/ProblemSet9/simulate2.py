def main(theta, params, kgrid, pi, V, PF, N, T, z_grid):
    import numpy as np
    import numba
    import zgrid2
    import z_markov2 as Z

    a_k, psi, rho, sigma, tao = theta

    betafirm, delta, mu, sizez, a_l = params

    # getting z shocks
    Zmat = np.zeros((N, T))
    for i in range(N):
        Zmat[i] = Z.sim_markov(z_grid, pi, T)

    Zdraw = Zmat.astype(dtype=np.int)

    # simulate k
    kloc_sim = np.zeros((N, T), dtype = np.int)
    for i in range(N):
        for j in range(T-1):
            kloc_sim[i, j+1] = PF[Zdraw[i, j]][kloc_sim[i, j]]
 
    k_sim = kgrid[kloc_sim]
    
    # simulate V
    V_sim = np.zeros((N, T))
    for i in range(N):
        for j in range(T):
            V_sim[i, j] = V[Zdraw[i, j]][kloc_sim[i, j]]
            
    # simulate pi   
    pi_sim = np.zeros((N, T))
    for i in range(N):
        for j in range(T):
            pi_sim[i, j] = z_grid[Zdraw[i, j]] * kgrid[kloc_sim[i, j]] ** a_k


    # simulate I
    I_sim = np.zeros((N, T))
    for i in range(N):
        for j in range(T-1):
            I_sim[i, j + 1] = k_sim[i, j+1] - k_sim[i, j] * (1-delta)
    
    # simulate ext finance
    tao_sim = np.zeros((N, T))
    tao_sim = I_sim / pi_sim - I_sim    

    # chopping up the first 100 observations
    k_sim = k_sim[:, -100:]
    I_sim = I_sim[:, -100:]
    V_sim = V_sim[:, -100:]
    pi_sim= pi_sim[:, -100:]
    tao_sim= tao_sim[:, -100:]
    
    return k_sim, I_sim, V_sim, pi_sim, tao_sim