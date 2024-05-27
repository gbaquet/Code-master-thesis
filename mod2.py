import math
import numpy as np
import matplotlib.pyplot as plt
from skfda.datasets import make_gaussian_process
from scipy import linalg
from scipy.optimize import minimize
import loc_lin_est
import time


#parameters#
n=100 ##sample size##
G=51 ##number of (equispaced) points##
a_l=0.4
b_l=0.6
h_grid=1/(G-1)
bandwidth=0.06

#functions#
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
#generating cov matrix to simulate process#
def cov_matrix(G):
    K=np.zeros((G,G))
    for i in range(G):
        for j in range(G):
            s=i*h_grid
            t=j*h_grid
            K[i,j]=cov(s,t)
    return K
#generating sample and stocking it in CSR format#
iX=[0]
jX=[]
X=[]
fragments=np.zeros(shape=(n,2))
count=0
l_bar=0.0 ##needed to compute the subset S##
K=cov_matrix(G)
for k in range(1,n+1):
    fd=make_gaussian_process(
        n_samples=1,
        n_features=G,
        start=0.0,
        stop=1.0,
        mean=0.0,
        cov=K,
        noise=0.0,
        random_state=None
    )
    l=np.random.uniform(a_l,b_l,1)[0]
    M=np.random.uniform(a_l/2,1-(a_l/2),1)[0]
    A=max(0,M-(l/2))
    B=min(1,M+(l/2))
    fragments[k-1,0]=A
    fragments[k-1,1]=B
    l_bar=l_bar+(B-A)
    t=math.ceil((A-math.floor(A))/h_grid)*h_grid+math.floor(A)
    T=math.floor((B-math.floor(B))/h_grid)*h_grid+math.floor(B)
    d=round((T-t)/h_grid)
    j=round(t/h_grid)
    for l in range(d):
        count=count+1
        jX.append(j)
        X.append(fd.data_matrix[0,j,0])
        j=j+1
    iX.append(count)

xX=np.array(X,dtype=np.float64)
jX=np.array(jX,dtype=np.int64)
iX=np.array(iX,dtype=np.int64)

#creating matrix which gives us dicretized S#
l_bar*=(1/n)
S=np.zeros((G,G),dtype=np.int64)
m=np.zeros((G,G))
for i in range(G):
    for j in range(G):
        if abs(i-j)*h_grid<=l_bar:
            S[i,j]=1
            
for i in range(n):
    for j in range(iX[i],iX[i+1]):
        for l in range(iX[i],iX[i+1]):
            m[jX[j],jX[l]]+=1

for i in range(G):
    for j in range(G):
        if m[i,j]>=min(10,math.ceil(m.max()/20)):
            S[i,j]=1
plt.imshow(S,origin='lower', extent=[0, 1, 0, 1])
plt.show()
p=math.floor(math.sqrt(np.sum(S)))-1
if p%2!=0:
    p-=1

#computing K_hat on S with optimal h based on real cov
for i in range(G):
    for j in range(G):
        if S[i,j]==0:
            K[i,j]=0.0

def score_h(hat,cov):
    score=hat-cov
    score=np.square(score)
    score=np.sum(score)
    return score

S=S.flatten()
score_vec=[]
band=[]

result_opt=loc_lin_est.K_hat_mat(G,h_grid, bandwidth,jX, iX, xX, n, S)
score_opt=score_h(result_opt,K)
score_vec.append(score_opt)
band.append(bandwidth)
bandwidth+=(h_grid/2)

while bandwidth<0.15:
    result=loc_lin_est.K_hat_mat(G,h_grid, bandwidth,jX, iX, xX, n, S)
    score=score_h(result,K)
    score_vec.append(score)
    band.append(bandwidth)
    if score<score_opt:
        result_opt=result
        bandwidth+=(h_grid/2)
    else:
        bandwidth+=(h_grid/2)
    

plt.scatter(band, score_vec)
plt.xlabel('Bandwidth')
plt.ylabel('Score to minimize')
plt.title('Optimal bandwidth')
plt.show()

S=np.reshape(S,(G,G))
count=0
N=round(np.sum(S))
scat=np.zeros((N,3))
for i in range(G):
    for j in range(G):
        if S[i,j]==1:
            scat[count,0]=i*h_grid
            scat[count,1]=j*h_grid
            scat[count,2]=result_opt[i,j]
            count+=1
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
scatter=ax.scatter(scat[:,0],scat[:,1],scat[:,2])
plt.show()

# creating gamma_D and W_D matrix #
'Fourier basis'
def fourier(x,j):
    if j==1:
        g=x
    elif j==2:
        g=1
    elif (j%2)==0 and j>2:
        g=math.sqrt(2)*math.cos(2*math.pi*x*((j-2)/2))
    elif (j%2)!=0 and j>2:
        g=math.sqrt(2)*math.sin(2*math.pi*x*((j-1)/2))
    return g

'Gram-Schmidt matrix for grid points'
def G_S(J):
    x=np.linspace(0.0,1.0,J)
    matrix=np.zeros((len(x),p))
    for i in range(p):
        for j in range(len(x)):
            matrix[j,i]=fourier(x[j],i+1)
    result,_=np.linalg.qr(matrix)
    return result

base=G_S(G)
        
'gamma matrix + W_D'
W_D=np.zeros((N,N))
GAMMA=np.zeros((N,p*p))
count=0
K_hat_vec=np.zeros((N,1))
K_hat_vec[:,0]=scat[:,2]

for i in range(G):
    for j in range(G):
        if S[i,j]==1:
            kron_prod=np.kron(base[i,:],base[j,:])
            GAMMA[count,:]=kron_prod
            if i==j:
                W_D[count,count]=G
            else:
                W_D[count,count]=1
            count+=1

# finding initial guess #
X_T_WX=GAMMA.T@W_D@GAMMA
print(np.linalg.cond(X_T_WX))
det=np.linalg.det(X_T_WX)
lamb=10**(-16)
if det==0:
    while det==0:
        X_T_WX=X_T_WX+lamb*np.eye(p*p)
        det=np.linalg.det(X_T_WX)
        lamb*=10
print(np.linalg.cond(X_T_WX))
X_T_WX_inv=np.linalg.inv(X_T_WX)
X_T_WK=GAMMA.T@W_D@K_hat_vec
result= X_T_WX_inv@X_T_WK
result=h_grid**2*result
result=np.reshape(result,(p,p))
for i in range(1,p):
    for j in range(i):
        result[i,j]=result[j,i]
eigenvalues,eigenvectors=np.linalg.eigh(result)
small_pos = np.min(eigenvalues[eigenvalues > 0])
eigenvalues[eigenvalues<0]=small_pos
eigval_matrix=np.diag(eigenvalues)
guess=eigenvectors@linalg.sqrtm(eigval_matrix)
guess=guess.ravel()
print("guess found")

# finding constraint solution #
'objective funnction'
def obj(b_p):
    b_p=np.reshape(b_p,(p,p))
    b_p=b_p@b_p.T
    b_p=np.reshape(b_p,(p*p,1))
    f1=K_hat_vec-GAMMA@b_p
    f=f1.T@W_D@f1
    f=f*h_grid**2
    return f

MATRIX=GAMMA.T@W_D
MATRIX=(2*h_grid**2)*MATRIX

solution=minimize(obj,guess)
sol=solution.x
print("solution found")
sol=np.reshape(sol,(p,p))
sol=sol@sol.T

# plotting estimated covariance #
def cov_hat_scalar(s,t):
    id_s=round(s/h_grid)
    id_t=round(t/h_grid)
    est=base[id_s].T@sol@base[id_t]
    return est
cov_hat=np.vectorize(cov_hat_scalar)

ax=plt.axes(projection="3d")
x_data=np.arange(0.0,1.0,h_grid)
y_data=np.arange(0.0,1.0,h_grid)
x,y=np.meshgrid(x_data,y_data)
z=cov_hat(x,y)
ax.plot_surface(x,y,z,cmap="plasma")
plt.show()

ax=plt.axes(projection="3d")
x_data=np.arange(0.0,1.0,h_grid)
y_data=np.arange(0.0,1.0,h_grid)
x,y=np.meshgrid(x_data,y_data)
z=cov(x,y)-cov_hat(x,y)
ax.plot_surface(x,y,z,cmap="plasma")
plt.show()