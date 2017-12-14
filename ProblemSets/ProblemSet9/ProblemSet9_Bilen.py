import numpy as np
import scipy.optimize as opt
from scipy.misc import derivative
import numba
import time

# setting up the parameters
betafirm = .95 
delta = 0.15 
psi = .164
sigma = 0.613 
mu = 0  
rho = 0.1605 
sizez = 9 
a_k = .690 
a_l = .650     
w=.90
tao = 0
params = (betafirm, delta, mu, sizez, a_l)

# setting up the number of firms and years to be simulated
N = 1000
T = 50


def Qfunc(theta):
    
    '''
    This is a function that calculates the distance between simulated moments 
    and the actual moments for the setup used for table 2
    '''
    
    import kgrid
    import zgrid
    import z_markov
    import simulate as sim
    import VFI
    
    # getting the grid space for capital
    kgrid, sizek = kgrid.main(params,w,a_k)

    # getting the grid space for shocks
    z_grid, pi = zgrid.main(sigma, mu, rho, sizez)

    # getting firm's optimal decision rule
    V, PF = VFI.main(theta, params, z_grid, kgrid, sizek, sizez, 
                                               pi, a_k, w, psi)
    
    # simulate the data 
    k_sim, I_sim, V_sim, pi_sim = sim.main(theta, params, kgrid, 
                                               pi, V, PF, N, T, z_grid)

   
    # define variables of interest
    I_K = (I_sim/k_sim).reshape((N*T, 1))
    Qav = (V_sim/k_sim).reshape((N*T, 1))
    PI_K = (pi_sim/k_sim).reshape((N*T, 1))
    
    # get the necessary statistics 
    I_Kr = I_K.reshape((1, N*T))
    sc = np.corrcoef(I_Kr[0][1:], I_Kr[0][:N*T-1])[0,1]
    sd = np.std(PI_K)
    qbar = V_sim.sum() / k_sim.sum()

    # run the q-regression
    Y = np.matrix(I_K)
    X = np.matrix(np.concatenate((np.ones((N*T, 1)), Qav, PI_K), axis = 1))
    a0, a1, a2 = np.linalg.inv(X.T * X) * X.T * Y 

    # construct the vectors
    mu_sim = np.array((a1[0, 0], a2[0, 0], sc, sd, qbar))
    mu_real = np.array((.03, .24, .4, .25, 3))
    
    # identity matrix
    Wmat = np.eye(len(mu_sim))

    # get the distance between simulated statistics and the actual statistics
    dist = np.matrix((mu_real - mu_sim)) * np.linalg.inv(Wmat) * np.matrix((mu_real - mu_sim)).T

    return dist

# construct the theta vector which will hold the results
theta = np.array([a_k, psi, rho, sigma])

# call the minimizer
results = opt.minimize(Qfunc, theta, method='Nelder-Mead', options={'maxiter': 2000})
thetahat=results['x']

# to get the standard errors, first define a func that returns the simulated moments
def get_sim(theta):
    
    '''
    This is a function that returns the simulated moments 
    to be used to obtain the standard errors
    '''
    
    import kgrid
    import zgrid
    import z_markov
    import simulate as sim
    import VFI
    
    # getting the grid space for capital
    kgrid, sizek = kgrid.main(params,w,a_k)

    # getting the grid space for shocks
    z_grid, pi = zgrid.main(sigma, mu, rho, sizez)

    # getting firm's optimal decision rule
    V, PF = VFI.main(theta, params, z_grid, kgrid, sizek, sizez, 
                                               pi, a_k, w, psi)
    
    # simulate the data 
    k_sim, I_sim, V_sim, pi_sim = sim.main(theta, params, kgrid, 
                                               pi, V, PF, N, T, z_grid)

   
    # define variables of interest
    I_K = (I_sim/k_sim).reshape((N*T, 1))
    Qav = (V_sim/k_sim).reshape((N*T, 1))
    PI_K = (pi_sim/k_sim).reshape((N*T, 1))
    
    # get the necessary statistics 
    I_Kr = I_K.reshape((1, N*T))
    sc = np.corrcoef(I_Kr[0][1:], I_Kr[0][:N*T-1])[0,1]
    sd = np.std(PI_K)
    qbar = V_sim.sum() / k_sim.sum()

    # run the q-regression
    Y = np.matrix(I_K)
    X = np.matrix(np.concatenate((np.ones((N*T, 1)), Qav, PI_K), axis = 1))
    a0, a1, a2 = np.linalg.inv(X.T * X) * X.T * Y 

    # construct the vectors
    mu_sim = np.array((a1[0, 0], a2[0, 0], sc, sd, qbar))
    
    return mu_sim

# get the standard errors
mu_sim = get_sim(thetahat)
std_err = np.diagonal(np.sqrt((1+1/N)*derivative(get_sim, thetahat) * np.linalg.inv(np.eye(len(mu_sim)))*
                              derivative(get_sim, thetahat)))

print("The results for table 2: " ,results['x'])
print("The standard errors for table 2: ", std_err)

###############################################################################################

def Qfunc2(theta):
    
    '''
    This is a function that calculates the distance between simulated moments 
    and the actual moments for the setup used for table 3
    '''
    
    import kgrid2
    import zgrid2
    import z_markov2
    import simulate2 as sim2
    import VFI2
    
    # getting the grid space for capital
    kgrid, sizek = kgrid2.main(params,w,a_k)

    # getting the grid space for shocks
    z_grid, pi = zgrid2.main(sigma, mu, rho, sizez)

    # getting firm's optimal decision rule
    V, PF = VFI2.main(theta2, params, z_grid, kgrid, sizek, sizez, 
                                               pi, a_k, w, psi)
    
    # simulate the data 
    k_sim, I_sim, V_sim, pi_sim, tao_sim = sim2.main(theta2, params, kgrid, 
                                               pi, V, PF, N, T, z_grid)

   
    # define variables of interest
    I_K = (I_sim/k_sim).reshape((N*T, 1))
    Qav = (V_sim/k_sim).reshape((N*T, 1))
    PI_K = (pi_sim/k_sim).reshape((N*T, 1))
    
    # get the necessary statistics 
    I_Kr = I_K.reshape((1, N*T))
    sc = np.corrcoef(I_Kr[0][1:], I_Kr[0][:N*T-1])[0,1]
    sd = np.std(PI_K)
    qbar = V_sim.sum() / k_sim.sum()
    taor = tao_sim.reshape((N*T, 1)).sum() / N*T

    # run the q-regression
    Y = np.matrix(I_K)
    X = np.matrix(np.concatenate((np.ones((N*T, 1)), Qav, PI_K), axis = 1))
    a0, a1, a2 = np.linalg.inv(X.T * X) * X.T * Y 

    # construct the vectors
    mu_sim = np.array((a1[0, 0], a2[0, 0], sc, sd, qbar, taor))
    mu_real = np.array((.03, .24, .4, .25, 3, .25))
    
    # identity matrix
    Wmat = np.eye(len(mu_sim))

    # get the distance between simulated statistics and the actual statistics
    dist = np.matrix((mu_real - mu_sim)) * np.linalg.inv(Wmat) * np.matrix((mu_real - mu_sim)).T

    return dist


# construct the theta vector which will hold the results
theta2 = np.array([a_k, psi, rho, sigma, tao])


# call the minimizer
results2 = opt.minimize(Qfunc2, theta2, method='Nelder-Mead', options={'maxiter': 2000})
thetahat2=results2['x']

# to get the standard errors, first define a func that returns the simulated moments
def get_sim2(theta2):
    
    '''
    This is a function that returns the simulated moments 
    to be used to obtain the standard errors
    '''
    
    import kgrid2
    import zgrid2
    import z_markov2
    import simulate2 as sim2
    import VFI2
    
    # getting the grid space for capital
    kgrid, sizek = kgrid2.main(params,w,a_k)

    # getting the grid space for shocks
    z_grid, pi = zgrid2.main(sigma, mu, rho, sizez)

    # getting firm's optimal decision rule
    V, PF = VFI2.main(theta2, params, z_grid, kgrid, sizek, sizez, 
                                               pi, a_k, w, psi)
    
    # simulate the data 
    k_sim, I_sim, V_sim, pi_sim, tao_sim = sim2.main(theta2, params, kgrid, 
                                               pi, V, PF, N, T, z_grid)

   
    # define variables of interest
    I_K = (I_sim/k_sim).reshape((N*T, 1))
    Qav = (V_sim/k_sim).reshape((N*T, 1))
    PI_K = (pi_sim/k_sim).reshape((N*T, 1))
    
    # get the necessary statistics 
    I_Kr = I_K.reshape((1, N*T))
    sc = np.corrcoef(I_Kr[0][1:], I_Kr[0][:N*T-1])[0,1]
    sd = np.std(PI_K)
    qbar = V_sim.sum() / k_sim.sum()
    taor = tao_sim.reshape((N*T, 1)).sum() / N*T

    # run the q-regression
    Y = np.matrix(I_K)
    X = np.matrix(np.concatenate((np.ones((N*T, 1)), Qav, PI_K), axis = 1))
    a0, a1, a2 = np.linalg.inv(X.T * X) * X.T * Y 

    # construct the vectors
    mu_sim = np.array((a1[0, 0], a2[0, 0], sc, sd, qbar, taor))
    
    return mu_sim

# get the standard errors
mu_sim2 = get_sim2(thetahat2)
std_err2 = np.diagonal(np.sqrt((1+1/N)*derivative(get_sim2, thetahat2) * np.linalg.inv(np.eye(len(mu_sim2)))*
                              derivative(get_sim2, thetahat2)))

print("The results for table 3: " ,results2['x'])
print("The standard errors for table 3: ", std_err2)

