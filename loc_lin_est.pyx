import numpy as np
cimport numpy as np

cdef double kernel(double bandwidth,int p,double u):
    cdef double w
    if abs(u/bandwidth)<=1:
        w=(1/bandwidth)*((u/bandwidth)**p)*(3/4)*(1-(u/bandwidth)**2)
    else:
        w=0.0
    return w


cdef double M(np.ndarray[np.int64_t, ndim=1] jx,np.ndarray[np.int64_t, ndim=1] ix,double h_grid, double bandwidth, int i, double x,int n):
    cdef double s
    s=0.0
    cdef int j,l
    cdef double k
    for j in range(n):
        k=0.0
        for l in range(ix[j],ix[j+1]):
            k=k+kernel(bandwidth,i,jx[l]*h_grid-x)
        s=s+k/(ix[j+1]-ix[j])
    s=s/n
    return s


cdef double S(np.ndarray[np.int64_t, ndim=1] jx,np.ndarray[np.int64_t, ndim=1] ix,np.ndarray[np.float64_t, ndim=1] ax, double h_grid, double bandwidth, int i, double x,int n):
    cdef double s
    s=0.0
    cdef int j,l
    cdef double k
    for j in range(n):
        k=0.0
        for l in range(ix[j],ix[j+1]):
            k=k+kernel(bandwidth,i,jx[l]*h_grid-x)*ax[l]
        s=s+k/(ix[j+1]-ix[j])
    s=s/n
    return s


cdef double Q(np.ndarray[np.int64_t, ndim=1] jx,np.ndarray[np.int64_t, ndim=1] ix,double h_grid, double bandwidth, int p_1,int p_2, double s,double t,int n):
    cdef double x
    x=0.0
    cdef int j,l,r
    cdef double k
    for j in range(n):
        k=0.0
        for l in range(ix[j],ix[j+1]):
            for r in range(ix[j],ix[j+1]):
                if l!=r:
                    k=k+kernel(bandwidth,p_1,jx[l]*h_grid-s)*kernel(bandwidth,p_2,jx[r]*h_grid-t)
        x=x+k/(ix[j+1]-ix[j])
    x=x/n
    return x


cdef double R(np.ndarray[np.int64_t, ndim=1] jx,np.ndarray[np.int64_t, ndim=1] ix,np.ndarray[np.float64_t, ndim=1] ax, double h_grid, double bandwidth, int p_1,int p_2, double s,double t,int n):
    cdef double x
    x=0.0
    cdef int j,l,r
    cdef double k
    for j in range(n):
        k=0.0
        for l in range(ix[j],ix[j+1]):
            for r in range(ix[j],ix[j+1]):
                if l!=r:
                    k=k+kernel(bandwidth,p_1,jx[l]*h_grid-s)*kernel(bandwidth,p_2,jx[r]*h_grid-t)*ax[l]*ax[r]
        x=x+k/(ix[j+1]-ix[j])
    x=x/n
    return x

##S_mat is the vectorised version of S
def K_hat_mat(int G,double h_grid,double bandwidth,np.ndarray[np.int64_t, ndim=1] jx,np.ndarray[np.int64_t, ndim=1] ix,np.ndarray[np.float64_t, ndim=1] ax, int n,np.ndarray[np.int64_t, ndim=1] S_mat):
    cdef double r,K
    cdef double q_20,q_02,q_11,q_10,q_01,q_00
    cdef double r_00,r_01,r_10
    cdef double m_1,m_2,m_3
    cdef double d
    cdef double s,t
    cdef double m_s,m_t
    cdef int i,j,count
    cdef double m
    cdef double s_0,s_1,m_0
    m_hat=np.zeros(G, dtype=np.float64)
    for i in range(G):
        t=i*h_grid
        m_0=M(jx,ix,h_grid,bandwidth,0, t,n)
        m_1=M(jx,ix,h_grid,bandwidth,1, t,n)
        m_2=M(jx,ix,h_grid,bandwidth,2, t,n)
        s_0=S(jx,ix,ax,h_grid,bandwidth,0, t,n)
        s_1=S(jx,ix,ax,h_grid,bandwidth,1, t,n)
        m=(s_0*m_2-s_1*m_1)/(m_0*m_2-m_1**2)
        m_hat[i]=m
    
    K_hat=np.zeros((G,G), dtype=np.float64)
    for i in range(G):
        for j in range(i+1):
            if S_mat[i*G+j]==1:
                s=i*h_grid
                t=j*h_grid
                q_20=Q(jx,ix,h_grid,bandwidth, 2,0,s,t,n)
                q_02=Q(jx,ix,h_grid,bandwidth, 0,2,s,t,n)
                q_11=Q(jx,ix,h_grid,bandwidth, 1,1,s,t,n)
                q_10=Q(jx,ix,h_grid,bandwidth, 1,0,s,t,n)
                q_01=Q(jx,ix,h_grid,bandwidth, 0,1,s,t,n)
                q_00=Q(jx,ix,h_grid,bandwidth, 0,0,s,t,n)
                r_00=R(jx,ix,ax,h_grid,bandwidth, 0,0,s,t,n)
                r_01=R(jx,ix,ax,h_grid,bandwidth, 0,1,s,t,n)
                r_10=R(jx,ix,ax,h_grid,bandwidth, 1,0,s,t,n)
                m_1=q_20*q_02-q_11**2
                m_2=q_10*q_02-q_01*q_11
                m_3=q_01*q_20-q_10*q_11
                d=m_1*q_00-m_2*q_10-m_3*q_01
                r=(m_1*r_00-m_2*r_10-m_3*r_01)/d
                m_s=m_hat[i]
                m_t=m_hat[j]
                K=r-m_s*m_t
                K_hat[i][j]=K
                K_hat[j][i]=K
                count=count+1
    return K_hat
