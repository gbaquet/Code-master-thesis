import math
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import opt_cv
import time

#parameters#
n=100 ##sample size##
G=26 ##number of (equispaced) points##
n_sim=50
grid_points=26
h_grid=1/(grid_points-1)

#covariance function#
def cov_scalar(s,t):
    K=0.0
    for i in range(1,51):
        if i%2==0:
            phi_s=math.cos(2*math.pi*s*((i+1)/2))
            phi_t=math.cos(2*math.pi*t*((i+1)/2))
            a=i**(-2)*2*phi_s*phi_t
            K+=a
        else:
            phi_s=math.sin(2*math.pi*s*((i+1)/2))
            phi_t=math.sin(2*math.pi*t*((i+1)/2))
            a=i**(-2)*2*phi_s*phi_t
            K+=a
    return K
cov=np.vectorize(cov_scalar)

diff=np.zeros(n_sim)

for j in range(n_sim):
    print("SIMULATION",j)
    #generating sample#
    data=np.zeros((n,G))
    t=np.zeros((n,G))
    for i in range(n):
        x_data = np.random.uniform(0, 1, G).reshape(-1,1)
        x_data = np.sort(x_data, axis=0)
        t[i]=x_data[:,0]
        x,y=np.meshgrid(x_data,x_data)
        K=cov(x,y)
        z=multivariate_normal.rvs(mean=np.zeros(G), cov=K, size=1)
        data[i]=z

    # computing and plotting real cov #
    x_data = np.linspace(0.0,1.0,grid_points)
    x, y = np.meshgrid(x_data, x_data)
    K=cov(x,y)

    # computing and plotting cov_est with optimal h based on CV #
    start=time.time()
    h_cv=opt_cv.cv_pyt(data, t, G, n, 10, 0.025) 
    end=time.time()
    print("Computation time:",(end-start)/60,"min")
    print()


    # computing and plotting cov_est with optimal h based on ISE (which uses real cov) #
    start=time.time()
    h_ise=opt_cv.ise_pyt(data, t, K, h_grid, grid_points, G, n, 0.025)
    end=time.time()
    print("Computation time:",(end-start)/60,"min")
    print()

    # adding diff between both bandwidths #
    diff[j]=h_ise-h_cv
    print("diff=",h_ise-h_cv)
    print()

# getting results #
df = pd.DataFrame(diff)
df.to_excel('output.xlsx', index=False, header=False)
