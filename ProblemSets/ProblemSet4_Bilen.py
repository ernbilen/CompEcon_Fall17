import pandas as pd
import numpy as np
import scipy.optimize as sc
from scipy.optimize import differential_evolution
import geopy
from geopy.distance import vincenty

# Reading in the data
data=pd.read_excel('radio_merger_data2.xlsx')

# Adjusting the price and the population as thousands
data['price'] = data['price'] / 1000
data['population_target'] = data['population_target'] / 1000

# A function that calculates the distance between the buyer and the target       
def dist_calc(d):
    buyer_loc = (d['buyer_lat'], d['buyer_long'])
    target_loc = (d['target_lat'], d['target_long'])
    return vincenty(buyer_loc, target_loc).miles

# Creating two different dataframes for each year
dset_2007 = data[data.year == 2007].reset_index(drop=True)
dset_2008 = data[data.year == 2008].reset_index(drop=True)

# Creating a dataset with counterfactual mergers
b_char = ['year', 'buyer_id', 'buyer_lat', 'buyer_long', 'num_stations_buyer','corp_owner_buyer']
t_char = ['target_id', 'target_lat', 'target_long', 'price', 'hhi_target', 'population_target']
dsets = [dset_2007, dset_2008]
counter_fact = [x[b_char].iloc[i].values.tolist() + x[t_char].iloc[j].values.tolist() for x in dsets for i in range(len(x)) for j in range(len(x)) if i != j]
counter_fact = pd.DataFrame(counter_fact, columns = b_char + t_char)

# Distance between real buyer and real target
data['distance'] = data.apply (lambda d: dist_calc(d),axis = 1)

# Distance between counterfactual buyer and counterfactual target
counter_fact['distance'] = counter_fact.apply (lambda d: dist_calc(d),axis = 1)

def score(coeffs):
    
    '''
    This function calculates the payoff functions inside the indication function. If f(b,t) + f(b',t') > + f(b't) + f(b,t')  holds,
    then the indicator equals 1, 0 otherwise.
    
    '''
    
    f_b_t = data['num_stations_buyer'] * data['population_target']+ coeffs[0] * data['corp_owner_buyer'] * data['population_target'] + coeffs[1] * data['distance']
    
    f_cb_ct = counter_fact['num_stations_buyer'] * counter_fact['population_target'] + coeffs[0] * counter_fact['corp_owner_buyer'] * counter_fact['population_target'] + coeffs[1] * counter_fact['distance']
  
    f_cb_t = counter_fact['num_stations_buyer'] * data['population_target'] + coeffs[0] * counter_fact['corp_owner_buyer'] * data['population_target'] + coeffs[1] * data['distance']
   
    f_b_ct = data['num_stations_buyer'] * counter_fact['population_target'] + coeffs[0] * data['corp_owner_buyer'] * counter_fact['population_target'] + coeffs[1] * data['distance']
    
 
    j = f_b_t + f_cb_ct
    k = f_cb_t + f_b_ct
    
    # Checking inequality,
    i=(j>k)
    
    # Summing the total and returning -total since we maximize our objective func by using a minimizer method 
    total=i.sum()
    return -total

bounds = [(-.5,.5),(-.5,.5)]
results = differential_evolution(score, bounds)
print(results.x)

def score2(coeffs):
    
    '''
    This function calculates the payoff functions inside the indication function. If f(b,t) + f(b',t') > + f(b't) + f(b,t')  holds,
    then the indicator equals 1, 0 otherwise.
    
    '''
    
    f_b_t = coeffs[0] * data['num_stations_buyer'] * data['population_target'] + coeffs[1] * data['corp_owner_buyer'] * data['population_target'] + coeffs[2] * data['hhi_target'] + coeffs[3] * data['distance']
    
    f_cb_ct = coeffs[0] * counter_fact['num_stations_buyer'] * counter_fact['population_target'] + coeffs[1] * counter_fact['corp_owner_buyer'] * counter_fact['population_target'] + coeffs[2]*counter_fact['hhi_target'] + coeffs[3] * counter_fact['distance']
  
    f_cb_t = coeffs[0] * counter_fact['num_stations_buyer'] * data['population_target'] + coeffs[1] * counter_fact['corp_owner_buyer'] * data['population_target'] + coeffs[2] * data['hhi_target'] + coeffs[3] * data['distance']
   
    f_b_ct = coeffs[0] * data['num_stations_buyer'] * counter_fact['population_target'] + coeffs[1] * data['corp_owner_buyer'] * counter_fact['population_target'] + coeffs[2] * data['hhi_target'] + coeffs[3] * data['distance']
    
 
    j = f_b_t + f_cb_ct
    k = f_cb_t + f_b_ct
    
    # Checking inequality,
    i=(j>k)
    
    # Summing the total and returning -total since we maximize our objective func by using a minimizer method 
    total=i.sum()
    return -total

bounds2 = [(-.5,.5),(-.5,.5),(-.5,.5),(-.5,.5)]
results2 = differential_evolution(score2, bounds2)
print(results2.x)