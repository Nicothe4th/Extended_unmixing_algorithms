#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:22:23 2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [P,A,Ds,S,Yh,Ji]=NEBEAE3(Y,N,parameters,Po,oae)
%
% Estimation of Optimal End-members and Abundances in Nonlinear Mixture
%  Model (Normalized Model)
%
% Based on --> Daniel U. Campos-Delgado et al. "Nonlinear Extended Blind End-member and 
% Abundance Extraction for Hyperspectral Images", Signal Processing, Vol. 201, 
% December 2022, pp. 108718, DOI: 10.1016/j.sigpro.2022.108718
%
%
% Input Arguments
%
%   Y = matrix of measurements (LxK)
%   N = order of multi-linear mixture model
%   parameters = 8x1 vector of hyper-parameters in BEAE methodology
%              = [initicond rho lambda epsilon maxiter downsampling  ...
%                      parallel display]
%       initcond = initialization of end-members matrix {1,...,8}
%                                 (1) Maximum cosine difference from mean
%                                      measurement (default)
%                                 (2) Maximum and minimum energy, and
%                                      largest distance from them
%                                 (3) PCA selection + Rectified Linear Unit
%                                 (4) ICA selection (FOBI) + Rectified
%                                 Linear Unit
%                                 (5) N-FINDR endmembers estimation in a 
%                                 multi/hyperspectral dataset (Winter,1999)
%                                 (6) Vertex Component Analysis (VCA)
%                                 (Nascimento and Dias, 2005)
%                                 (7) Simplex Volume Maximization (SVMAX) (Chan et
%                                 al. 2011)
%                                 (8) Simplex identification via split augmented 
%                                  Lagrangian (SISAL) (Bioucas-Dias, 2009)
%       rho = regularization weight in end-member estimation 
%             (default rho=0.1);
%       lambda = entropy weight in abundance estimation \in [0,1) 
%                (default lambda=0);
%       epsilon = threshold for convergence in ALS method 
%                 (default epsilon=1e-3); 
%       maxiter = maximum number of iterations in ALS method
%                 (default maxiter=20);
%       downsampling = percentage of random downsampling in end-member 
%                      estimation [0,1) (default downsampling=0.5);
%       parallel = implement parallel computation of abundances (0 -> NO or 1 -> YES)
%                  (default parallel=0);
%       display = show progress of iterative optimization process (0 -> NO or 1 -> YES)
%                 (default display=0);
%   Po = initial end-member matrix (LxN)
%   oae = only optimal abundance estimation with Po (0 -> NO or 1 -> YES)
%         (default oae = 0)
%
% Output Arguments
%
%   P --> matrix of end-members (LxN)
%   A --> abudances matrix (NxK)
%   Ds --> vector of nonlinear interaction levels (Kx1)
%   S  --> scaling vector (Kx1)
%   Yh --> estimated matrix of measurements (LxK) --> 
%
%       AA=(A.*repmat(S',[N,1]))
%       Yh=repmat((1-Ds)',[L,1]).*(P*AA) + repmat(Ds',[L,1]).*((P*AA).*Y)
%
%   Ji --> vector with cost function values during iterative process
%
% Daniel Ulises Campos Delgado
% FC-UASLP & ULPGC
% Version: July/2024
@author: jnmc
"""
import numpy as np
from scipy.sparse.linalg import svds
import scipy.linalg as splin


from scipy.linalg import  pinv, eigh
from scipy.linalg import svd, eig, sqrtm
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

def initPo(Yo, Ym, initcond, N):
    """
    Inicializa la matriz Po según diferentes condiciones iniciales.

    Args:
        Yo (numpy.ndarray): Datos de entrada (LxM).
        Ym (numpy.ndarray): Matriz modificada (LxM).
        initcond (int): Condición inicial a usar.
        N (int): Número de vectores a estimar.

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
    return Po

def abundance(Z, Y, P, D, Lambda, parallel=False):
    """
    Estimación de Abundancias Óptimas en un Modelo de Mezcla No Lineal.

    Args:
        Z (numpy.ndarray): Matriz de medidas.
        Y (numpy.ndarray): Matriz de medidas normalizadas.
        P (numpy.ndarray): Matriz de endmembers.
        D (numpy.ndarray): Vector de probabilidades de mezcla no lineal.
        V (numpy.ndarray): Matriz de parámetros adicionales.
        Lambda (float): Peso de la entropía en la estimación de abundancia (0,1).
        parallel (bool): Si es True, ejecuta en paralelo.

    Returns:
        numpy.ndarray: Matriz de abundancias estimadas.
    """
    L, K = Y.shape
    N = P.shape[1]
    A = np.zeros((N, K))
    em = np.eye(N)

    if P.shape[0] != L:
        raise ValueError("ERROR: the number of rows in Y and P does not match")

    def compute_abundances(k):
        """Función auxiliar para calcular las abundancias."""
        
        yk = np.squeeze(np.c_[Y[:, k]])
        zk = np.squeeze(np.c_[Z[:, k]])
        byk = np.dot(Y[:, k], Y[:, k])
        dk = D[k].copy()
        deltakn = (1 - dk) * np.ones(N) + dk * np.dot(P.T, zk)
        Pk = P * ((1 - dk) + dk * zk[:, None])
        bk = np.dot(Pk.T, yk)
        Go = np.dot(Pk.T, Pk)
        lmin = np.min(np.linalg.eigvals(Go))
        G = Go - em * lmin * Lambda
        Gi = np.linalg.inv(G)

        # Solución óptima no restringida
        sigma = (np.dot(deltakn.T, np.dot(Gi, bk)) - 1) / np.dot(deltakn.T, np.dot(Gi, deltakn))
        ak = np.dot(Gi, (bk - deltakn * sigma))

        # Verificar elementos negativos
        if np.sum(ak >= 0) != N:
            Iset = np.zeros(N, dtype=bool)
            while np.sum(ak < 0) != 0:
                Iset[ak < 0] = True
                Ll = np.sum(Iset)
                Q = N + 1 + Ll
                Gamma = np.zeros((Q, Q))
                Beta = np.zeros(Q)

                Gamma[:N, :N] = G.copy()
                Gamma[:N, N] = deltakn*byk
                Gamma[N, :N] = deltakn.T.copy()

                cont = 0
                for i in range(N):
                    if Iset[i]:
                        cont += 1
                        Gamma[i, N + cont] = 1
                        Gamma[N + cont, i] = 1

                Beta[:N] = bk.copy()
                Beta[N] = 1 
                delta = np.linalg.solve(Gamma, Beta)
                ak = delta[:N].copy()
                ak[np.abs(ak) < 1e-9] = 0

        return ak

    # Implementación paralela o secuencial
    if parallel:
        A = np.array(Parallel(n_jobs=-1)(delayed(compute_abundances)(k) for k in range(K))).T
    else:
        for k in range(K):
            A[:, k] = compute_abundances(k)

    return A

def probanonlinear(Z, Y, P, A, parallel=False):
    """
    Estimación de la probabilidad de mezclas no lineales.

    Args:
        Z (numpy.ndarray): Matriz de medidas.
        Y (numpy.ndarray): Matriz de medidas normalizadas (sum(axis=0) = 0).
        P (numpy.ndarray): Matriz de endmembers (sum(axis=0) = 0).
        A (numpy.ndarray): Matriz de abundancias.
        V (numpy.ndarray): Matriz de parámetros adicionales.
        parallel (bool): Si es True, ejecuta en paralelo.

    Returns:
        numpy.ndarray: Vector de probabilidades de mezclas no lineales ([-inf, 1)).
    """
    K = Y.shape[1]
    D = np.zeros(K)

    def compute_probability(k):
        yk = np.squeeze(Y[:, k])  # sk = Y(:, k) - V(:, k)
        zk = np.squeeze(Z[:, k])           # zk = Z(:, k)
        ak = np.squeeze(A[:, k])          # ak = A(:, k)
        ek = np.dot(P, ak)      # ek = P * ak
        T1 = ek - yk            # T1 = ek - sk
        T2 = ek - (ek * zk)     # T2 = ek - ek .* zk (element-wise product)
        
        # Calcula dk
        numerator = np.dot(T1.T, T2)  # T1' * T2 (producto escalar)
        denominator = np.dot(T2.T, T2)  # T2' * T2 (producto escalar)
        dk = min(1, numerator / denominator)
        return dk

    if parallel:
        D = np.array(Parallel(n_jobs=-1)(delayed(compute_probability)(k) for k in range(K)))
    else:
        for k in range(K):
            D[k] = compute_probability(k)

    return D




def compute_grad_pk(Y_col,  zk, ak, dk, Po, L, rho, byk):
    yk = Y_col
    Mk = np.diag((1 - dk) * np.ones(L) + dk * zk)
    GradPK = - np.dot(Mk.T, yk[:, np.newaxis] @ ak[np.newaxis, :])/ byk + (Mk.T @ Mk) @ Po @ (ak[:, np.newaxis] @ ak[np.newaxis, :]) / byk
    return GradPK

def compute_num_den(Y_col, Z_col, A_col, D_val, Po, GradP, L, rho, byk):
    onesL = np.ones(L)
    
    yk = Y_col 
    zk = Z_col
    ak = A_col
    dk = D_val
    
    Mk = np.diag((1 - dk) * onesL + dk * zk)
    T1 = Mk @ GradP @ ak
    numG = T1.T @ Mk @ (Po @ ak - yk) / byk
    denG = T1.T @ T1 / byk
    return numG, denG

def endmember(Z, Y, Po, A, D, rho, parallel):
    N, K = A.shape
    L = Y.shape[0]
    R = np.sum(N - np.arange(1, N))
    em = np.eye(N)

    if parallel == 1:
        GradPK = Parallel(n_jobs=-1)(delayed(compute_grad_pk)(
            Y[:, k].copy(),  Z[:, k].copy(), 
            A[:, k].copy(), D[k], Po, L, rho, 
            Y[:, k].T @ Y[:, k]) for k in range(K))
        GradP = np.sum(GradPK, axis=0)
    else:
        GradPK = [compute_grad_pk(
            Y[:, k].copy(), Z[:, k].copy(), 
            A[:, k].copy(), D[k], Po, L, rho, 
            Y[:, k].T @ Y[:, k]) for k in range(K)]
        GradP = np.sum(GradPK, axis=0)

    O = N * em - np.ones((N, N))
    GradP = GradP / K + rho * Po @ O / R

    # Compute Optimal Step in Update Rule
    if parallel == 1:
        results = Parallel(n_jobs=-1)(delayed(compute_num_den)(
            Y[:, k].copy(), Z[:, k].copy(), 
            A[:, k].copy(), D[k], Po, GradP, L, rho, 
            Y[:, k].T @ Y[:, k]) for k in range(K))
        numG = sum([res[0] for res in results]) + rho * np.trace(GradP @ O @ Po.T + Po @ O @ GradP.T) / R / 2
        denG = sum([res[1] for res in results]) + rho * np.trace(GradP @ O @ GradP.T) / R
    else:
        results = [compute_num_den(
            Y[:, k].copy(), Z[:, k].copy(), 
            A[:, k].copy(), D[k], Po, GradP, L, rho, 
            Y[:, k].T @ Y[:, k]) for k in range(K)]
        numG = sum([res[0] for res in results]) + rho * np.trace(GradP @ O @ Po.T + Po @ O @ GradP.T) / R / 2
        denG = sum([res[1] for res in results]) + rho * np.trace(GradP @ O @ GradP.T) / R
    
    
    if denG == 0:
        alpha=0
    else:
        alpha = max(0, numG / denG)

    # Compute the Steepest Descent Update of End-members Matrix
    P_est = Po - alpha * GradP
    P_est[P_est < 0] = 0
    P_est[np.isnan(P_est)] = 0
    P_est[np.isinf(P_est)] = 0
    P = P_est / np.sum(P_est, axis=0)

    return P

def nebeae(Yo=[], N=2, parameters=[], Po=[], oae=0):
    initcond=1;
    rho=0.1;
    Lambda=0;
    epsilon=1e-3;
    maxiter=20;
    downsampling=0.5;
    parallel=0;
    display=0;

    if len(parameters) != 8:
        print('The length of parameters vector is not 8 !!')
        print('Default values of hyper-parameters are used instead')
    else:
        initcond=int(parameters[0])
        rho = parameters[1]
        Lambda = parameters[2]
        epsilon=parameters[3]
        maxiter=parameters[4]
        downsampling=parameters[5]
        parallel=parameters[6]
        display=parameters[7]         
    
    if initcond != 1 and initcond != 2 and initcond != 3  and initcond != 4 and initcond != 5 and initcond != 6 and initcond != 7 and initcond != 8:
        print("The initialization procedure of endmembers matrix is 1,2,3,4,5,6,7 or 8")
        print("The default value is considered!")
        initcond = 1;    
    if rho<0:
        print('The regularization weight rho cannot be negative');
        print('The default value is considered!');
        rho=0.1;
        
    if Lambda<0 or Lambda>=1:
        print('The entropy weight lambda is limited to [0,1)');
        print('The default value is considered!');
        Lambda=0;    

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
        else:
            if Po.shape[0]!=Yo.shape[0]:
                print('The number of spectral channels in Po is not correct !')
                print('The initialization is considered by VCA of the input dataset');
                initcond=6
            elif Po.shape[1]>N:
                print('The number of columns in Po is larger than the order of the linear model!!');
                print('The initialization is considered by VCA of the input dataset');
                initcond=6
            else:
                initcond=0
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
     
    
    ## Normalización
    mYm = np.sum(Y, axis=0)
    mYmo = np.sum(Yo, axis=0)
    Ym = Y / mYm[np.newaxis, :]  # np.newaxis para mantener las dimensiones
    Ymo = Yo / mYmo[np.newaxis, :]
    NYm = np.linalg.norm(Ym, 'fro')  # Norma Frobenius
    

    """
    Selection of Initial End-members Matrix

    """
    if initcond>0:
        if display==1:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
            print('Estimating initial conditions of free end-members');
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        Po = initPo(Yo,Ym,initcond,N)   
    else:
        if display==1:
            print('The end-members matrix Po provided by the user!')
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
    iter=0;
    J=1e5;
    Jp=1e6;
    Ji=[]
    Dm=np.zeros((K,1));
    
    if display==1:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
            print('NEBEAE, Multi-Linear Unmixing');
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
    
    while ((Jp-J)/Jp) >= epsilon and iter < maxiter and oae==0:
        
        Am = abundance(Y,Ym,P,Dm,Lambda,parallel);
        Dm = probanonlinear(Y,Ym,P,Am,parallel);    

        Pp = P.copy();
        P = endmember(Y,Ym,Pp,Am,Dm,rho,parallel); 
        
        Jp=J;
        residual = Ym - ((1 - Dm)[np.newaxis, :] * (P @ Am)) - (Dm[np.newaxis, :] * ((P @ Am) * Y)) 
        J = np.linalg.norm(residual, "fro")
        Ji.append((Jp - J) / Jp)
        if J>Jp:
            P=Pp.copy()
            if display == 1:
                print(f"Break Number of iteration = {iter}")
                print(f"Breck Percentage Estimation Error = {100 * J / NYm} %")
            break
        if display == 1:
            print(f"Number of iteration = {iter}")
            print(f"Percentage Estimation Error = {100 * J / NYm} %")
        iter += 1
        
   
    if downsampling == 0:
        A = Am.copy();
        D = Dm.copy();
    else:
        Ins = np.sort(list(set(I) - set(Is)))
        Ins = np.array(Ins, dtype=int)
        J = 1e5
        Jp = 1e6
        Dms = np.zeros(len(Ins))
        iter = 1
        Ymos = np.c_[Ymo[:, Ins]]
        Yos = np.c_[Yo[:, Ins]]
        
        while ((Jp-J)/Jp) >= epsilon and iter<=maxiter:
            Ams = abundance(Yos, Ymos, P, Dms, Lambda, parallel)
            Dms = probanonlinear(Yos, Ymos, P, Ams, parallel)
            Jp = J
            residual = Ymos - np.multiply((1 - Dms).reshape(1, -1), P @ Ams) - \
                       np.multiply(Dms.reshape(1, -1), np.multiply(P @ Ams, Yos))
            J = np.linalg.norm(residual, 'fro')
            iter += 1
            
        A = np.hstack((Am, Ams))
        D = np.hstack((Dm, Dms))
        II = np.hstack((Is, Ins))
    
        # Sort and reorder
        Index = np.argsort(II)  # Get the indices for sorting
        A = A[:, Index]
        D = D[Index]
    S=mYmo
    AA = A * mYmo
    
    Yh = (np.expand_dims(1-D, axis=0) * (P @ AA) + np.expand_dims(D, axis=0) * ((P @ AA) * Yo))   
    Ds=probanonlinear(Yo,Yo,P,A,parallel);
    
    return P,A,Ds,S,Yh,Ji