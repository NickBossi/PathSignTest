#Random Walk 
#Nicholas Bossi
#03 March 2024

#Defines a function which will create a random walk over multivariate normally distributed random variables of dimension m 
# over n time-steps


import numpy as np

#Gobal variables of number of BVN's and the correlation used
n= 100
m = 2
pho = 0.5

def random_walk(m,n,pho, mean = 0, sd = 1):
    #Generating the standard normals
    std_normals = np.random.normal(mean, sd, size = (2,n))

    #Creating the covariance matrix
    sigma_Cov = np.full((m,m), pho)
    np.fill_diagonal(sigma_Cov, 1)

    #Finding A
    A = np.linalg.cholesky(sigma_Cov)

    #Creating our final, correlated normals
    norms = A @ std_normals

    #Creating the random walk
    walk = np.zeros((m,n))
    walk[:,0] = norms[:,0]

    for i in range(n-1):
        walk[:,i+1] = walk[:,i]+norms[:,i+1]
    
    return(walk)

ran_walk = random_walk(2,100,0.5)
print(ran_walk)