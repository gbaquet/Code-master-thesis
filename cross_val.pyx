import numpy as np
cimport numpy as np

cdef double kernel(double bandwidth,int p,double u):
    cdef double w
    if abs(u/bandwidth)<=1:
        w=(1/bandwidth)*((u/bandwidth)**p)*(3/4)*(1-(u/bandwidth)**2)
    else:
        w=0.0
    return w


cdef np.ndarray[np.float64_t, ndim=2] cov(np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.float64_t, ndim=2] t, double h_grid, int grid_points, int G, int n, int k_sets, double incr):
    cdef int i,j,l,k,z,f,q,b
    cdef double bandwidth,result,score1,score2,h,u1,u2,s1,s2
    cdef int obs_per_set
    cdef double M_1,M_2,M_3
    cdef double q_20,q_02,q_11,q_10,q_01,q_00
    cdef double r_00,r_01,r_10
    cdef double k10,k11,k12,k20,k21,k22
    cdef double d
    cdef double k0,k1,k2,y
    obs_per_set=round(n/k_sets)
    score2=10.0**23
    bandwidth=0.0
    # computing optimal h #
    for i in range(100000):
        score1=0.0
        bandwidth+=incr
        print("For bandwidth",bandwidth,":")
        for k in range(k_sets):
            for j in range(n):
                if j<k*obs_per_set or j>=(k+1)*obs_per_set:
                    continue
                for f in range(G):
                    for q in range(f):
                        #building estimate for (T_jf,T_jq)#
                        if f==q:
                            continue
                        q_20=0.0
                        q_02=0.0
                        q_11=0.0
                        q_10=0.0
                        q_01=0.0
                        q_00=0.0
                        r_00=0.0
                        r_10=0.0
                        r_01=0.0
                        for l in range(n):
                            if k*obs_per_set<=l<(k+1)*obs_per_set:
                                continue
                            for z in range(G):
                                for b in range(G):
                                    if z==b:
                                        continue
                                    u1=t[j,f]-t[l,z]
                                    u2=t[j,q]-t[l,b]
                                    k10=kernel(bandwidth,0,u1)
                                    k11=kernel(bandwidth,1,u1)
                                    k12=kernel(bandwidth,2,u1)
                                    k20=kernel(bandwidth,0,u2)
                                    k21=kernel(bandwidth,1,u2)
                                    k22=kernel(bandwidth,2,u2)
                                    y=data[l,z]*data[l,b]
                                    q_00+=k10*k20
                                    q_10+=k11*k20
                                    q_01+=k10*k21
                                    q_11+=k11*k21
                                    q_20+=k12*k20
                                    q_02+=k10*k22
                                    r_00+=k10*k20*y
                                    r_10+=k11*k20*y
                                    r_01+=k10*k21*y
                        M_1=q_20*q_02-q_11**2
                        M_2=q_10*q_02-q_01*q_11
                        M_3=q_01*q_20-q_10*q_11
                        d=M_1*q_00-M_2*q_10-M_3*q_01
                        if d!=0.0:
                            result=(M_1*r_00-M_2*r_10-M_3*r_01)/d
                        else:
                            score1+=10.0**23
                            result=0.0
                        #testing estimate on test data#
                        score1+=(data[j,f]*data[j,q]-result)**2 
        print("score=",score1)
        print()       
        if score1<score2:
            score2=score1
            h=bandwidth
        else:
            break
    # computing E[Y(t)Y(s)] with optimal h #
    print("Optimal CV bandwidth:",h)
    print()
    bandwidth=h
    cov=np.zeros((grid_points,grid_points), dtype=np.float64)
    for j in range(grid_points):
        for i in range(j+1):
            s1=j*h_grid
            s2=i*h_grid
            q_20=0.0
            q_02=0.0
            q_11=0.0
            q_10=0.0
            q_01=0.0
            q_00=0.0
            r_00=0.0
            r_10=0.0
            r_01=0.0
            for l in range(n):
                for z in range(G):
                    for b in range(G):
                        if z==b:
                            continue
                        u1=s1-t[l,z]
                        u2=s2-t[l,b]
                        k10=kernel(bandwidth,0,u1)
                        k11=kernel(bandwidth,1,u1)
                        k12=kernel(bandwidth,2,u1)
                        k20=kernel(bandwidth,0,u2)
                        k21=kernel(bandwidth,1,u2)
                        k22=kernel(bandwidth,2,u2)
                        y=data[l,z]*data[l,b]
                        q_00+=k10*k20
                        q_10+=k11*k20
                        q_01+=k10*k21
                        q_11+=k11*k21
                        q_20+=k12*k20
                        q_02+=k10*k22
                        r_00+=k10*k20*y
                        r_10+=k11*k20*y
                        r_01+=k10*k21*y
            M_1=q_20*q_02-q_11**2
            M_2=q_10*q_02-q_01*q_11
            M_3=q_01*q_20-q_10*q_11
            d=M_1*q_00-M_2*q_10-M_3*q_01
            result=(M_1*r_00-M_2*r_10-M_3*r_01)/d
            cov[j,i]=result
            cov[i,j]=result
    return cov

def cov2(np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.float64_t, ndim=2] t, double h_grid, int grid_points, int G, int n, int k_sets, double incr):
    par=np.zeros((grid_points,grid_points), dtype=np.float64)
    par=cov(data, t, h_grid, grid_points, G, n, k_sets, incr)
    return par

cdef np.ndarray[np.float64_t, ndim=2] cov3(np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.float64_t, ndim=2] t, np.ndarray[np.float64_t, ndim=2] K, double h_grid, int grid_points, int G, int n, double incr):
    cdef int i,j,l,k,z,b
    cdef double bandwidth,result,score1,score2,h,u1,u2,s1,s2
    cdef double M_1,M_2,M_3
    cdef double q_20,q_02,q_11,q_10,q_01,q_00
    cdef double r_00,r_01,r_10
    cdef double k10,k11,k12,k20,k21,k22
    cdef double d
    cdef double y
    score2=10.0**23
    bandwidth=0.0
    # computing E[Y(t)Y(s)] with optimal h #
    covar1=np.zeros((grid_points,grid_points), dtype=np.float64)
    covar2=np.zeros((grid_points,grid_points), dtype=np.float64)
    mat=np.zeros((grid_points,grid_points), dtype=np.float64)
    for k in range(100000):
        score1=0.0
        bandwidth+=incr
        print("For bandwidth",bandwidth,":")
        for j in range(grid_points):
            for i in range(j+1):
                s1=j*h_grid
                s2=i*h_grid
                q_20=0.0
                q_02=0.0
                q_11=0.0
                q_10=0.0
                q_01=0.0
                q_00=0.0
                r_00=0.0
                r_10=0.0
                r_01=0.0
                for l in range(n):
                    for z in range(G):
                        for b in range(G):
                            if z==b:
                                continue
                            u1=s1-t[l,z]
                            u2=s2-t[l,b]
                            k10=kernel(bandwidth,0,u1)
                            k11=kernel(bandwidth,1,u1)
                            k12=kernel(bandwidth,2,u1)
                            k20=kernel(bandwidth,0,u2)
                            k21=kernel(bandwidth,1,u2)
                            k22=kernel(bandwidth,2,u2)
                            y=data[l,z]*data[l,b]
                            q_00+=k10*k20
                            q_10+=k11*k20
                            q_01+=k10*k21
                            q_11+=k11*k21
                            q_20+=k12*k20
                            q_02+=k10*k22
                            r_00+=k10*k20*y
                            r_10+=k11*k20*y
                            r_01+=k10*k21*y
                M_1=q_20*q_02-q_11**2
                M_2=q_10*q_02-q_01*q_11
                M_3=q_01*q_20-q_10*q_11
                d=M_1*q_00-M_2*q_10-M_3*q_01
                result=(M_1*r_00-M_2*r_10-M_3*r_01)/d
                covar1[j,i]=result
                covar1[i,j]=result
        mat=K-covar1
        mat=mat*mat
        score1=np.sum(mat)
        print("score=",score1)
        print()
        if score1<score2:
            score2=score1
            for i in range(grid_points):
                for j in range(i+1):
                    covar2[i,j]=covar1[i,j]
                    covar2[j,i]=covar1[i,j]
            h=bandwidth
        else:
            break
    print("Optimal ISE bandwidth:",h)
    return covar2

def cov4(np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.float64_t, ndim=2] t, np.ndarray[np.float64_t, ndim=2] K, double h_grid, int grid_points, int G, int n, double incr):
    par=np.zeros((grid_points,grid_points), dtype=np.float64)
    par=cov3(data, t, K, h_grid, grid_points, G, n, incr)
    return par