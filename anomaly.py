
import numpy as np

#training set
#x: {n features, m samples as n*m array}
#x =  np.array([ […] ... ])
#for e.g., a 3 * 4 array for 3 fearures, 4 samples

x = np.array( [ [1,2,4,5], 
                [3,4,1,3],
                [4,5,3,5] ])

#calculate u (mean) and var (variance)
u = np.mean(x,axis=1,keepdims=True)
var = np.mean((x - u)**2, axis=1, keepdims=True)

#gaussian probability (vectorised implementation)
# x {n features, m samples as n*m array}
# u = mean of each feature as n * 1 array
# var = variance of each feature as n* 1 array
def prob_gaussian(x,u,var):
 p1 = 1/np.sqrt(2*np.pi*var)
 p2 = np.exp(- ((x-u)**2) / (2*var))
 return p1*p2

#epsilon - anomaly threshold, to be tuned based on performance on anomalous vs normal samples
epsilon = 0.02

#flags anomaly if prob_prod<epsilon
# x_test {n features, m samples as n*m array}
# u = mean of each feature as n * 1 array
# var = variance of each feature as n* 1 array
# epsilon = anomaly threshold
def anomaly(x_test, u, var, epsilon):
 probs = prob_gaussian(x_test, u, var)
 prob_prod = np.prod(probs,axis=0)
 return prob_prod, prob_prod<epsilon

#test with some examples

x_test = [[3],
          [2],
          [4]]

anomaly(x_test, u, var, epsilon)
#(array([0.03351253]), array([False]))

x_test = [[0],
          [5],
          [6]]

#(array([9.39835262e-05]), array([ True]))
