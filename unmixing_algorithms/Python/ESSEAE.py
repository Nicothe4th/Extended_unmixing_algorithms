#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:50:59 2024

@author: jnmc
"""


import numpy as np
from scipy.sparse.linalg import svds
import scipy.linalg as splin


from scipy.linalg import  pinv, eigh
from scipy.linalg import svd,  sqrtm
from joblib import Parallel, delayed



def pca(X,d):
    L, N = X.shape
    xMean = X.mean(axis=1).reshape((L,1),order='F')
    xzm = xMean - np.tile(xMean, (1, N))
    U, _ , _  = svds( (xzm @ xzm.T)/N , k=d)
    return U
def NFINDR(Y, N):
    """
    N-FINDR endmembers estimation in multi/hyperspectral dataset
    """
    L, K = Y.shape
    # dimention redution by PCA
    U = pca(Y,N)
    Yr = U.T @ Y
    # Initialization
    Po = np.zeros((L,N))
    IDX = np.zeros((1,K))
    TestM = np.zeros((N,N))
    TestM[0,:]=1
    for i in range (N):
        idx = np.floor(float(np.random.rand(1))*K) + 1
        TestM[1:N,i]= Yr[:N-1,i].copy()
        IDX[0,i]=idx
    actualVolume = np.abs(np.linalg.det(TestM))
    it=1
    v1=-1
    v2=actualVolume
    #  Algorithm
    maxit = 3 * N
    while (it<maxit and v2>v1):
        for k in range (N):
            for i in range (K):
                actualSample = TestM[1:N,k].copy()
                TestM[1:N,k] = Yr[:N-1,i].copy()
                volume = np.abs(np.linalg.det(TestM))
                if volume > actualVolume:
                    actualVolume = volume.copy()
                    IDX[0,k] = i
                else:
                    TestM[1:N,k]=actualSample.copy()
        it = it + 1
        v1 = v2
        v2 = actualVolume.copy()
    
    for i in range (N):
        Po[:,i] = Y[:,int(IDX[0,i])].copy()
    return Po

def vca(Y,R):
    #############################################
    # Initializations
    #############################################
    [L, N]=Y.shape   # L number of bands (channels), N number of pixels     
    R = int(R)
    #############################################
    # SNR Estimates
    #############################################  
    y_m = np.mean(Y,axis=1,keepdims=True)
    Y_o = Y - y_m           # data with zero-mean
    Ud  = splin.svd(np.dot(Y_o,Y_o.T)/float(N))[0][:,:R]  # computes the R-projection matrix 
    x_p = np.dot(Ud.T, Y_o)                 # project the zero-mean data onto p-subspace
    P_y     = np.sum(Y**2)/float(N)
    P_x     = np.sum(x_p**2)/float(N) + np.sum(y_m**2)
    SNR = 10*np.log10( (P_x - R/L*P_y)/(P_y - P_x) ) 
    SNR_th = 15 + 10*np.log10(R)+8
    
    #############################################
    # Choosing Projective Projection or 
    #          projection to p-1 subspace
    #############################################
    if SNR < SNR_th:
        d = R-1
        Ud = Ud[:,:d].copy()
        Yp =  np.dot(Ud,x_p[:d,:]) + y_m      # again in dimension L
        x = x_p[:d,:].copy() #  x_p =  Ud.T * Y_o is on a R-dim subspace
        c = np.argmax(np.sum(x**2,axis=0))**0.5
        y = np.vstack(( x, c*np.ones((1,N))))
    else:
        d=R
        Ud  = splin.svd(np.dot(Y,Y.T)/float(N))[0][:,:d] # computes the p-projection matrix 
        x_p = np.dot(Ud.T,Y)
        Yp =  np.dot(Ud,x_p[:d,:])      # again in dimension L (note that x_p has no null mean)
        x =  np.dot(Ud.T,Y)
        u =  np.mean(x,axis=1,keepdims=True)        #equivalent to  u = Ud.T * r_m
        y =  x / np.dot(u.T,x)
    #############################################
    # VCA algorithm
    #############################################
            
    indice = np.zeros((R),dtype=int)
    A = np.zeros((R,R))
    A[-1,0] = 1

    for i in range(R):
        w = np.random.rand(R,1);   
        f = w - np.dot(A,np.dot(splin.pinv(A),w))
        f = f / splin.norm(f)
        v = np.dot(f.T,y)
        indice[i] = np.argmax(np.abs(v))
        A[:,i] = y[:,indice[i]].copy()        # same as x(:,indice(i))
    Ae = Yp[:,indice].copy()
    Ae[Ae < 0] = 0
    return Ae

def SVMAX(X, N):
    """
    An implementation of SVMAX algorithm for endmember estimation.
    
    Parameters:
    X : numpy.ndarray
        Data matrix where each column represents a pixel and each row represents a spectral band.
    N : int
        Number of endmembers to estimate.
        
    Returns:
    A_est : numpy.ndarray
        Estimated endmember signatures (or mixing matrix).
    """
    M, L = X.shape
    d = np.mean(X, axis=1, keepdims=True)
    U = X - np.outer(d, np.ones((1, L)))
    C = eigh(U @ U.T, lower=False, subset_by_index=(M-N+1, M-1))[1]
    Xd_t = C.T @ U;
    
    A_set = np.zeros((N,N))
    
    index = np.zeros(3); 
    P = np.eye(N);    
    
                         
    Xd = np.vstack((Xd_t, np.ones((1, L))))
    
    
    for i in range(N):
        ind = np.argmax((np.sum((abs(P@Xd))**2,axis=0,keepdims=True))**(1/2));
        A_set[:,i] = Xd[:, ind]
        P = np.eye(N) - A_set @ pinv(A_set);
        index[i] = ind;
    Po = C @ Xd_t[:,index.astype(int)]+d @np.ones((1,N));
    
    
    return Po

def spectral_angle_mapper(v1, v2):
    cos_dist = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    return np.degrees(np.arccos(cos_dist))

def replace_em(Pfix, Po):
    Nf = Pfix.shape[1]
    N = Po.shape[1]
    dsam = np.zeros(N)

    for i in range(N):
        for j in range(Nf):
            sam = spectral_angle_mapper(Po[:, i], Pfix[:, j])
            dsam[i] += sam

    sorted_idx = np.argsort(dsam)[::-1]
    Pfar = Po[:, sorted_idx[:(N - Nf)]]
    Pnew = np.hstack((Pfix, Pfar))
    return Pnew
    


def initPo(Yo, Ym, initcond, N, Pf):
    """
    Inicializa la matriz Po según diferentes condiciones iniciales.

    Args:
        Yo (numpy.ndarray): Datos de entrada (LxM).
        Ym (numpy.ndarray): Matriz modificada (LxM).
        initcond (int): Condición inicial a usar.
        N (int): Número de vectores a estimar.
        Pf (numpy.ndarray): Matriz fija (LxNf).

    Returns:
        numpy.ndarray: Matriz Po inicializada (LxN).
    """
    L = Yo.shape[0]
    Po = np.zeros((L, N))

    if initcond in [1, 2]:
        if initcond == 1:
            index = 0
            pmax = np.mean(Yo, axis=1)
            Yt = Yo
            Po[:, index] = pmax
        elif initcond == 2:
            index = 0
            Y1m = np.sum(np.abs(Yo), axis=0)
            Imax = np.argmax(Y1m)
            Imin = np.argmin(Y1m)
            pmax = Yo[:, Imax]
            pmin = Yo[:, Imin]
            indices = np.arange(Yo.shape[1])
            Yt = Yo[:, np.setdiff1d(indices, [Imax, Imin])]
            Po[:, index] = pmax
            index += 1
            Po[:, index] = pmin
        while index < N - 1:
            ymax = np.zeros(index + 1)
            Imax = np.zeros(index + 1, dtype=int)
            for i in range(index + 1):
                e1m = np.sum(Yt * Po[:, i:i+1], axis=0) / (
                    np.sqrt(np.sum(Yt**2, axis=0)) * np.sqrt(np.sum(Po[:, i]**2))
                )
                ymax[i], Imax[i] = min((abs(val), idx) for idx, val in enumerate(e1m))
            Immax = np.argmin(ymax)
            IImax = Imax[Immax]
            pmax = Yt[:, IImax]
            index += 1
            Po[:, index] = pmax
            indices = np.arange(Yt.shape[1])
            Yt = Yt[:, np.setdiff1d(indices, [IImax])]
    elif initcond == 3:
        _, _, VV = svd(Ym.T, full_matrices=False)
        W = VV[:N, :].T
        Po = W * np.sign(W.T @ np.ones((L, 1))).T
    elif initcond == 4:
        Yom = np.mean(Ym, axis=1, keepdims=True)
        Yon = Ym - Yom
        _, S, VV = svd(Yon.T, full_matrices=False)
        Yo_w = pinv(sqrtm(np.diag(S))) @ VV @ Ym.T
        V, _, _ = svd((Yo_w**2).sum(axis=1, keepdims=True) * Yo_w @ Yo_w.T)
        W = VV.T @ sqrtm(np.diag(S)) @ V[:N, :].T
        Po = W * np.sign(W.T @ np.ones((L, 1))).T
    elif initcond == 5:
        Po = NFINDR(Ym, N)
    elif initcond == 6:
        Po = vca(Ym, N)
    elif initcond == 7:
        Po = SVMAX(Ym, N)
    else:
        print("The selection of initial condition is incorrect!")
        print("VCA is adopted by default")
        Po = vca(Ym, N)

    Po = replace_em(Pf, Po)
    return Po

def abundance(Y,P,V,Lambda,parallel):
    L, K = Y.shape
    N = P.shape[1]
    A = np.zeros((N, K))
    if P.shape[0] != L:
        raise ValueError("ERROR: the number of rows in Y and P does not match")
        
    # Compute fixed vectors and matrices
    c = np.ones((N, 1))
    d = 1.0
    Go = P.T @ P
    em = np.eye(N)
    lmin = np.min(np.linalg.eigvals(Go))
    G = Go - em * lmin * Lambda    
    
    while np.linalg.cond(G) > 1e6:
        Lambda /= 2
        G = Go - em * lmin * Lambda
        if Lambda < 1e-6:
            raise ValueError("Unstable numerical results in abundances estimation, update lambda!!")

    Gi = np.linalg.inv(G)
    T1 = Gi @ c
    T2 = c.T @ T1
    
    def process_column(k):
        vk = V[:, [k]].copy()  # Ensure a column vector
        yk = Y[:, [k]].copy()  # Ensure a column vector
        sk = yk - vk
        bk = P.T @ sk
        byk = float(yk.T @ yk)

        # Compute optimal unconstrained solution
        dk = (bk.T @ T1 - (1 - np.sum(vk))) / T2
        ak = Gi @ (bk - dk * c)
        
        # Check for negative elements
        if np.any(ak < 0):
            Iset = np.zeros((N, 1), dtype=bool)
            while np.any(ak < 0):
                Iset[ak < 0] = True
                Ll = np.sum(Iset)
                Q = N + 1 + Ll
                Gamma = np.zeros((Q, Q), dtype=np.float32)
                Beta = np.zeros((Q, 1), dtype=np.float32)

                Gamma[:N, :N] = G / byk
                Gamma[:N, N] = c.copy().ravel()
                Gamma[N, :N] = c.T.copy().ravel()

                cont = 0
                for i in range(N):
                    if Iset[i, 0]:
                        cont += 1
                        ind = N + cont
                        Gamma[i, ind] = 1
                        Gamma[ind, i] = 1

                Beta[:N] = bk / byk
                Beta[N] = d-sum(vk)
                delta = np.linalg.solve(Gamma, Beta)
                ak = delta[:N].copy()
                ak[np.abs(ak) < 1e-9] = 0

        return ak
    # Start computation of abundances
    if parallel:
        results = Parallel(n_jobs=-1)(delayed(process_column)(k) for k in range(K))
        A = np.hstack(results)
    else:
        for k in range(K):
            A[:, [k]] = process_column(k)

    return A

def hybridEndmember(Y,A,P,V,Nf,rho):
    N, K = A.shape
    L = Y.shape[0]
    Nu = N - Nf
    R = np.sum(N - np.arange(1, N))  # Equivalent to sum(N - (1:(N-1)))
    W = np.tile((1.0 / K / np.sum(Y**2, axis=0)), (Nu, 1)).T
    
    if Y.shape[1] != K:
        raise ValueError("ERROR: the number of columns in Y and A does not match")
        
    Pf = np.c_[P[:, :Nf]]
    Af = np.c_[A[:Nf, :]]
    Au = np.c_[A[Nf:N, :]]
    X = Y - Pf @ Af - V
    O1 = -2 * np.ones((Nu, Nf))
    O2 = N * np.eye(Nu) - np.ones((Nu, Nu))
    LU1 = np.ones((L, Nu))
    # Construct Optimal End-members Matrix
    T0 = Au @ (W * Au.T) + rho * O2.T / R
    while np.linalg.cond(T0) > 1e6:
        rho /= 10
        T0 = Au @ (W * Au.T) + rho * O2.T / R
        if rho < 1e-6:
            raise ValueError("Unstable numerical results in end-members estimation, update rho!!")
    
    V_matrix = np.linalg.inv(T0)
    T2 = X @ (W * Au.T) - ((rho / (2 * R)) * Pf @ O1.T)
    T1 = np.eye(L) - (1 / L) * np.ones((L, L))
    T3 = (1 / L) * LU1
    P_est = T1 @ T2 @ V_matrix + T3
    P_est[P_est < 0] = 0
    P_est[np.isnan(P_est)] = 0
    P_est[np.isinf(P_est)] = 0

    # Normalize Optimal Solution
    Pnn = np.hstack([Pf, P_est])
    Pn = Pnn / np.sum(Pnn, axis=0, keepdims=True)
    return Pn

def sparsenoise(Y, P, A, lm):
    # Compute the residual matrix
    E = Y - P @ A
    
    # Dimensions of Y
    L = Y.shape[0]
    
    # Compute Ye as the sum of squared Y, repeated across rows
    Ye = np.tile(np.sum(Y**2, axis=0), (L, 1))
    
    # Compute sparse noise matrix
    V = np.sign(E) * np.maximum(0, np.abs(E) - lm * Ye)
    
    # Ensure V is non-negative
    V = np.maximum(0, V)
    
    return V

def esseae(Yo=[], N=2, parameters=[], Po=[]):

    initcond = 1
    rho = 0.1
    Lambda = 0
    lm = 0.01
    epsilon = 1e-3
    maxiter = 20
    downsampling = 0.5
    parallel = 0
    display = 0
    Nf = 0
    
    global NUMERROR
    NUMERROR = 0

    # Check the size of parameters
    if len(parameters) != 9:
        print('The length of parameters vector is not 9 !!')
        print('Default values of hyper-parameters are used instead')
    else:
        initcond=int(parameters[0])
        rho = parameters[1]
        Lambda = parameters[2]
        lm=parameters[3]
        epsilon=parameters[4]
        maxiter=parameters[5]
        downsampling=parameters[6]
        parallel=parameters[7]
        display=parameters[8]         
    
    if initcond != 1 and initcond != 2 and initcond != 3  and initcond != 4 and initcond != 5 and initcond != 6 and initcond != 7 and initcond != 8:
        print("The initialization procedure of endmembers matrix is 1,2,3,4,5,6 or 7!")
        print("The default value is considered!")
        initcond = 1    
    if rho<0:
        print('The regularization weight rho cannot be negative');
        print('The default value is considered!');
        rho=0.1;
        
    if Lambda<0 or Lambda>=1:
        print('The entropy weight lambda is limited to [0,1)');
        print('The default value is considered!');
        Lambda=0;
        
    if lm<0:
        print('The sparse noise weight has to be positive');
        print('The default value is considered!');
        lm=0.01;

    if epsilon<0 or epsilon>0.5:
        print('The threshold epsilon cannot be negative or >0.5');
        print('The default value is considered!');
        epsilon=1e-3;

    if maxiter<0 and maxiter<100:
        print('The upper bound maxiter cannot be negative or >100');
        print('The default value is considered!');
        maxiter=20;

    if downsampling<0 and downsampling>1:
        print('The downsampling factor cannot be negative or >1');
        print('The default value is considered!');
        downsampling=0.5;

    if parallel!=0 and parallel!=1:
        print('The parallelization parameter is 0 or 1');
        print('The default value is considered!');
        parallel=0;

    if display!=0 and display!=1:
        print('The display parameter is 0 or 1');
        print('The default value is considered!');
        display=0;

    if N<2:
        print('The order of the linear mixture model has to greater than 2!');
        print('The default value N=2 is considered!');
        N=2;
    N=int(N);
    
    
    if len(Po):
        if type(Po) != np.ndarray:
            print("The initial end-members Po must be a matrix !!")
            print("The initialization is considered by VCA")
            initcond = 6
            Nf = 0
        else: 
            if Po.shape[0]==Yo.shape[0] and Po.shape[1]==N:
                initcond=0
                Nf=0
            elif Po.shape[0]!=Yo.shape[0]:
                print('The number of spectral channels in Po is not correct !')
                print('The initialization is considered by VCA of the input dataset');
                initcond=6
                Nf=0
            elif Po.shape[0]==Yo.shape[0] and Po.shape[1]<N:
                Nf=Po.shape[1]
                Pf = Po
            elif Po.shape[1]>N:
                print('The number of columns in Po is larger than the order of the linear model!!');
                print('The initialization is considered by VCA of the input dataset');
                initcond=6
                Nf=0
         
    ## Ramdom Downsampling
    if not isinstance(Yo, np.ndarray):
        raise ValueError("The measurements matrix Yo must be a NumPy array.")

    L, K = Yo.shape
    if L > K:
        raise ValueError("The number of spatial measurements must be larger than the number of time samples!")

    # Índices para reducción aleatoria
    I = list(range(0, K))
    Kdown = int(np.round(K * (1 - downsampling)))
    Is = np.random.choice(K, Kdown, replace=False)
    Is = np.sort(Is)
    
    Y = np.c_[Yo[:, Is]] # Submatriz reducida
    Vo = np.zeros((L, K))
    Vm = np.c_[Vo[:, Is]]  # Submatriz inicializada con ceros para los índices seleccionados            
    
    ## Normalización
    mYm = np.sum(Y, axis=0)
    mYmo = np.sum(Yo, axis=0)
    Ym = Y / mYm[np.newaxis, :]  # np.newaxis para mantener las dimensiones
    Ymo = Yo / mYmo[np.newaxis, :]
    NYm = np.linalg.norm(Ym, 'fro')  # Norma Frobenius

    """
    Selection of Initial End-members Matrix

    """
    if initcond>0 and Nf<N:
        if display==1:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
            print('Estimating initial conditions of free end-members');
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        Po = initPo(Yo,Ym,initcond,N,Pf)   
    else:
        if display==1:
            print('The end-members matrix Po is assumed fixed!')
    Po = np.where(Po < 0, 0, Po)
    Po = np.where(np.isnan(Po), 0, Po)
    Po = np.where(np.isinf(Po), 0, Po)
    mPo=Po.sum(axis=0,keepdims=True)
    P=Po/np.tile(mPo,(L,1))               
    
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Alternated Least Squares Procedure
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    
    iter=1;
    J=1e5;
    Jp=1e6;
    Ji=[]
    

    if display==1:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
            print('Hybrid EBEAE with Linear Unmixing & Sparse Noise');
            print(f'Model Order = {N}');
            if initcond==0:
                print('The end-members matrix is initialized externally by matrix Po');
            elif initcond==1:
                print('Po is constructed based on the maximum cosine difference from mean measurement'); 
            elif  initcond==2:
                print('Po is constructed based on the maximum and minimum energy, and largest difference from them');
            elif initcond==3:
                print('Po is constructed based on the PCA selection + Rectified Linear Unit');
            elif initcond==4:
                print('Po is constructed based on the ICA selection (FOBI) + Rectified Linear Unit');
            elif initcond==5:
                print('Po is constructed based on N-FINDR endmembers estimation by Winter (1999)');
            elif initcond==6:
                print('Po is constructed based on Vertex Component Analysis by Nascimento and Dias (2005)');
            elif initcond==7:
                print('Po is constructed based on Simplex Volume Maximization (SVMAX) (Chan et al. 2011)');
                
    while ((Jp-J)/Jp) >= epsilon and iter<=maxiter  and NUMERROR==0:
        Am = abundance(Ym,P,Vm,Lambda,parallel)
        Pp = P.copy()
        P = hybridEndmember(Ym,Am,P,Vm,Nf,rho)
        Vm = sparsenoise(Ym, P, Am, lm)
        Jp = J
        residual = Ym-P@Am-Vm
        J = np.linalg.norm(residual, "fro")
        Ji.append((Jp - J) / Jp)
        if J>Jp:
            P=Pp.copy()
            if display == 1:
                print(f"Number of iteration = {iter}")
                print(f"Percentage Estimation Error = {100 * J / NYm} %")
            break
        if display == 1:
            print(f"Number of iteration = {iter}")
            print(f"Percentage Estimation Error = {100 * J / NYm} %")
        iter += 1          
                
    if downsampling == 0:
        A = Am.copy();
        V = Vm.copy();
    else:
        Ins = np.sort(list(set(I) - set(Is)))
        Ins = np.array(Ins, dtype=int)
        J = 1e5
        Jp = 1e6
        iter = 1
        Ymos = np.c_[Ymo[:, Ins]]

        Vms = np.zeros_like(Ymos)
        while ((Jp-J)/Jp) >= epsilon and iter<=maxiter:
            Ams = abundance(Ymos, P, Vms, Lambda, parallel)
            Vms = sparsenoise(Ymos, P, Ams, lm)
            Jp = J
            residual = Ymos - P@Ams -Vms
            J = np.linalg.norm(residual, 'fro')
            iter += 1
        
        A = np.hstack((Am, Ams))
        V = np.hstack((Vm, Vms))
        II = np.hstack((Is, Ins))
    
        # Sort and reorder
        Index = np.argsort(II)  # Get the indices for sorting
        A = A[:, Index]
        V = V[:, Index]
    
    S=mYmo
    AA = A * mYmo
    V = V * mYmo
    
    Yh = (P @ AA) + V
    return P,A,S,Yh,V,Ji             
