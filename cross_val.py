import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import cross_val
import time

#parameters#
n=50 ##sample size##
G=26 ## r_1=...=r_n=G ##
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

#plotting N first samples#
N=10
for i in range(N):
    plt.plot(t[i],data[i],'--')
plt.title("Simulated paths")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.show()

# computing and plotting real cov #
x_data = np.linspace(0.0,1.0,grid_points)
x, y = np.meshgrid(x_data, x_data)
K=cov(x,y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, K, cmap='plasma')
ax.set_zlim(-2.5, 2.5)  
ax.set_zticks(np.arange(-2, 3, 1))
ax.set_xlabel('s')
ax.set_ylabel('t')
ax.set_zlabel('Cov')
ax.view_init(elev=20, azim=190)
plt.show()

# computing and plotting cov_est with optimal h based on CV #
start=time.time()
covariance=cross_val.cov2(data, t, h_grid, grid_points, G, n,10, 0.025) 
end=time.time()
print("Computation time:",(end-start)/60,"min")
print()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, covariance, cmap='plasma')
ax.set_zlim(-2.5, 2.5)  
ax.set_zticks(np.arange(-2, 3, 1))
ax.set_xlabel('s')
ax.set_ylabel('t')
ax.set_zlabel('Cov')
ax.view_init(elev=20, azim=190)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, K-covariance, cmap='plasma')
ax.set_zlim(-2.5, 2.5)  
ax.set_zticks(np.arange(-2, 3, 1))
ax.set_xlabel('s')
ax.set_ylabel('t')
ax.set_zlabel('Cov')
ax.view_init(elev=20, azim=190)
plt.show()


# computing and plotting cov_est with optimal h based on ISE (which uses real cov) #
start=time.time()
covariance=cross_val.cov4(data, t, K, h_grid, grid_points, G, n, 0.025)
end=time.time()
print("Computation time:",(end-start)/60,"min")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, covariance, cmap='plasma')
ax.set_zlim(-2.5, 2.5)  
ax.set_zticks(np.arange(-2, 3, 1)) 
ax.set_xlabel('s')
ax.set_ylabel('t')
ax.set_zlabel('Cov')
ax.view_init(elev=20, azim=190)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, K-covariance, cmap='plasma')
ax.set_zlim(-2.5, 2.5)  
ax.set_zticks(np.arange(-2, 3, 1)) 
ax.set_xlabel('s')
ax.set_ylabel('t')
ax.set_zlabel('Cov')
ax.view_init(elev=20, azim=190)
plt.show()

