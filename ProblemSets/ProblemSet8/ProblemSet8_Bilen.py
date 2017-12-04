import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
import numba
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# setting up the parameters
a_k = .297  # capital's share of output
a_l = .650  # labor share of output
delta = .154  # depreciation rate
psi = 1.080  # coefficient on quadratic adjustment costs
w = .7  # wage
r = .040  # interest rate
sigma = .213  # std dev of disturbances to z
mu = 0  # mean of ln(z) process
rho = .7605  # Persistance of z process
sizez = 9  # number of grid points in z space
betafirm = 1 / (1 + r)  # firm's discount factor
h = 6.616 # scaling parameter for disutility of labor

# calculate the grid for k
dens = 5
# put in bounds here for the capital stock space
kstar = ((((1 / betafirm - 1 + delta) * ((w / a_l) ** (a_l / (1 - a_l)))) /
             (a_k * (1 ** (1 / (1 - a_l))))) **
             ((1 - a_l) / (a_k + a_l - 1)))
kbar = 2*kstar
lb_k = 0.001
ub_k = kbar
krat = np.log(lb_k / ub_k)
numb = np.ceil(krat / np.log(1 - delta))
K = np.zeros(int(numb * dens))

for j in range(int(numb * dens)):
    K[j] = ub_k * (1 - delta) ** (j / dens)
kgrid = K[::-1]
sizek = kgrid.shape[0]

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


@numba.jit
def SD_loop(PF, pi, Gamma, sizez, sizek):
    '''
    This function consists of a loop to get the stationary 
    distribution using the numba method
    '''
    HGamma = np.zeros((sizez, sizek))
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            for m in range(sizez):  # loop over z'
                HGamma[m, PF[i, j]] = HGamma[m, PF[i, j]] + pi[i, m] * Gamma[i, j]
    return HGamma



def grand(w):

    '''
    This is a grand loop that first calculates the firm's optimal decision rule, the HH's 
    consumption and labor choices, then aggregates individual choice to the overall economy, and finally 
    calculates the distance between aggregate labor demand and supply
    '''

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
                e[i, j, k] = (op[i,j] - kgrid[k] + ((1 - delta) * kgrid[j]) -
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

    start_time = time.clock()
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V    
        Vmat = VFI_loop(V, e, betafirm, sizez, sizek, Vmat, pi)
        Vstore[:, :, VFiter] = V.reshape(sizez, sizek,) 
        V = Vmat.max(axis=2) 
        PF = np.argmax(Vmat, axis=2) 
        Vstore[:,:, i] = V  
        VFdist = (np.absolute(V - TV)).max()  
        VFiter += 1
    
    VFI_time = time.clock() - start_time
    if VFiter < VFmaxiter:
        print('Value function converged after this many iterations:', VFiter)
    else:
        print('Value function did not converge')
    print('VFI took ', VFI_time, ' seconds to solve')  
    VF = V

    # optimal capital stock k'
    optK = kgrid[PF]     

    # stationary distribution
    Gamma = np.ones((sizez, sizek)) * (1 / (sizek * sizez))
    SDtol = 1e-12
    SDdist = 7
    SDiter = 0
    SDmaxiter = 1000
    while SDdist > SDtol and SDmaxiter > SDiter:
        HGamma = SD_loop(PF, pi, Gamma, sizez, sizek)
        SDdist = (np.absolute(HGamma - Gamma)).max()
        Gamma = HGamma
        SDiter += 1
    if SDiter < SDmaxiter:
        print('Stationary distribution converged after this many iterations: ',SDiter)
    else:
        print('Stationary distribution did not converge')

    # optimal investment for an individual firm
    opti = optK - (1 - delta) * kgrid 
    
    # labor demand for an individual firm
    ld = np.zeros((sizez, sizek))
    for i in range(sizez):
        for j in range(sizek):
            ld[i,j] = (((a_l / w) ** (1 / (1 - a_l))) *
          ((kgrid[j] ** a_k) ** (1 / (1 - a_l))) * (z_grid[i] ** (1/(1 - a_l))))

    # adjustment costs
    adj_cost = psi / 2 * np.multiply((opti)**2, 1 / kgrid) 

    # output per firm
    y = (np.multiply(np.multiply((ld) ** a_l, kgrid ** a_k), np.transpose([z_grid])))

    # aggregate labor demand
    LD = np.multiply(ld, Gamma).sum() 

    # aggregate investment
    I = np.multiply(opti, Gamma).sum()  

    # aggregate adjustment costs
    ADJ = np.multiply(adj_cost, Gamma).sum()  

    # aggregate output
    Y = np.multiply(y, Gamma).sum()  

    # aggregate consumption
    C = Y - I - ADJ  

    # aggregate labor supply
    LS = w / (h * C)  

    # distance between aggregate labor demand and aggregate labor supply
    dist = abs(LD - LS) 

    print('|Ld-Ls|:', dist)
    print('wage:', w)
       
    return dist

# guess for wage
wage_guess = .7

# call the minimizer
results = opt.minimize(grand, wage_guess, method='Nelder-Mead', tol = 1e-8,
                       options={'maxiter': 5000})
# eq'm wage
wstar = results.x[0]
print('The equilibrium wage rate:', wstar)


def equil_solver(w):

    '''
    This is a function that calculates the optimal policy
    for k' and the stationary distribution for a given wage
    and is used to get the equilibrium solution
    '''

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
                e[i, j, k] = (op[i,j] - kgrid[k] + ((1 - delta) * kgrid[j]) -
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

    start_time = time.clock()
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V    
        Vmat = VFI_loop(V, e, betafirm, sizez, sizek, Vmat, pi)
        Vstore[:, :, VFiter] = V.reshape(sizez, sizek,) 
        V = Vmat.max(axis=2) 
        PF = np.argmax(Vmat, axis=2) 
        Vstore[:,:, i] = V  
        VFdist = (np.absolute(V - TV)).max()
        VFiter += 1
    
    VFI_time = time.clock() - start_time
    if VFiter < VFmaxiter:
        print('Value function converged after this many iterations:', VFiter)
    else:
        print('Value function did not converge')
    print('VFI took ', VFI_time, ' seconds to solve')  
    VF = V

    # optimal capital stock k'
    optK = kgrid[PF]     

    # stationary distribution
    Gamma = np.ones((sizez, sizek)) * (1 / (sizek * sizez))
    SDtol = 1e-12
    SDdist = 7
    SDiter = 0
    SDmaxiter = 1000
    while SDdist > SDtol and SDmaxiter > SDiter:
        HGamma = SD_loop(PF, pi, Gamma, sizez, sizek)
        SDdist = (np.absolute(HGamma - Gamma)).max()
        Gamma = HGamma
        SDiter += 1
    if SDiter < SDmaxiter:
        print('Stationary distribution converged after this many iterations: ',
              SDiter)
    else:
        print('Stationary distribution did not converge')
        
    return optK, Gamma

# get the optimal policy for k' and the stationary distribution
(optK, Gamma) = equil_solver(wstar)

# plot the stationary distribution over k
fig, ax = plt.subplots()
ax.plot(kgrid, Gamma.sum(axis=0))
plt.xlabel('Size of Capital Stock')
plt.ylabel('Density')
plt.title('Stationary Distribution over Capital')
plt.show()

# plot the policy function
fig, ax = plt.subplots()
ax.plot(kgrid, kgrid, '-', label='45 degree line',color="red",linewidth=2)
for i in range(9):
    ax.plot(kgrid, optK[i,:], '--', label='Capital Next Period')
plt.xlabel('Size of Capital Stock')
plt.ylabel('Optimal Choice of Capital Next Period')
plt.title('Policy Functions for Different Productivity Shocks')
ax.legend(['45 degree line'])
plt.show()

# stationary distribution in 3D
zmat, kmat = np.meshgrid(kgrid, np.log(z_grid))
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmat, zmat, Gamma, rstride=1, cstride=1, cmap=cm.Blues,
                    linewidth=0, antialiased=False)
ax.view_init(elev=20., azim=20) 
ax.set_xlabel(r'Log Productivity')
ax.set_ylabel(r'Capital Stock')
ax.set_zlabel(r'Density')
plt.title("Stationary Distribution in 3D")
plt.show()