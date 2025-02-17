#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:51:55 2024

@author: jnmc
"""

import time
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

def abundance(Z, Y, P, D, V, lambd, parallel=False):
    """
    Estimación de Abundancias Óptimas en un Modelo de Mezcla No Lineal.

    Args:
        Z (numpy.ndarray): Matriz de medidas.
        Y (numpy.ndarray): Matriz de medidas normalizadas.
        P (numpy.ndarray): Matriz de endmembers.
        D (numpy.ndarray): Vector de probabilidades de mezcla no lineal.
        V (numpy.ndarray): Matriz de parámetros adicionales.
        lambd (float): Peso de la entropía en la estimación de abundancia (0,1).
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
        vk = np.squeeze(np.c_[V[:, k]])
        sk = Y[:, k] - vk
        zk = np.squeeze(Z[:, k]).copy()
        byk = np.dot(Y[:, k], Y[:, k])
        dk = D[k].copy()
        deltakn = (1 - dk) * np.ones(N) + dk * np.dot(P.T, zk)
        Pk = P * ((1 - dk) + dk * zk[:, None])
        bk = np.dot(Pk.T, sk)
        Go = np.dot(Pk.T, Pk)
        eGo = np.real(eig(Go, right=False))
        eGo[np.isnan(eGo)] = 1e6
        eGo[np.isinf(eGo)] = 1e6
        lmin = np.min(eGo)
        G = Go - em * lmin * lambd
        Gi = np.linalg.inv(G)

        # Solución óptima no restringida
        sigma = (np.dot(deltakn.T, np.dot(Gi, bk)) - (1 - np.sum(vk))) / np.dot(deltakn.T, np.dot(Gi, deltakn))
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
                Gamma[:N, N] = (deltakn*byk).copy()
                Gamma[N, :N] = deltakn.copy()

                cont = 0
                for i in range(N):
                    if Iset[i]:
                        cont += 1
                        Gamma[i, N + cont] = 1
                        Gamma[N + cont, i] = 1

                Beta[:N] = bk.copy()
                Beta[N] = 1 - np.sum(vk)
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

def probanonlinear(Z, Y, P, A, V, parallel=False):
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
        sk = Y[:, k] - V[:, k]  # sk = Y(:, k) - V(:, k)
        zk = np.squeeze(Z[:, k]).copy()            # zk = Z(:, k)
        ak = np.squeeze(A[:, k]).copy()           # ak = A(:, k)
        ek = np.dot(P, ak)      # ek = P * ak
        T1 = ek - sk            # T1 = ek - sk
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

def hybrid_endmember(Z, Y, Po, A, D, V, Nf, rho, parallel=False):
    """
    Estimación de Endmembers Óptimos en un Modelo de Mezcla Lineal.

    Args:
        Z (numpy.ndarray): Matriz de medidas.
        Y (numpy.ndarray): Matriz de medidas normalizadas.
        Po (numpy.ndarray): Matriz de endmembers inicial.
        A (numpy.ndarray): Matriz de abundancias.
        D (numpy.ndarray): Vector de niveles de interacción no lineal.
        V (numpy.ndarray): Matriz de ruido disperso.
        Nf (int): Número de endmembers fijos.
        rho (float): Factor de ponderación para la regularización.
        parallel (bool): Si es True, utiliza procesamiento en paralelo.

    Returns:
        numpy.ndarray: Nueva matriz de endmembers.
    """
    N, K = A.shape
    L = Y.shape[0]
    Nu = N - Nf
    R = np.sum(N - np.arange(1, N))
    onesL = np.ones(L)

    Pf = np.c_[Po[:, :Nf]]
    Pu = np.c_[Po[:, Nf:]]
    Af = np.c_[A[:Nf, :]]
    Au = np.c_[A[Nf:, :]]
    
    # Computar X
    X = Y - ((1 - D).reshape(1, -1) * (Pf @ Af)) - (D.reshape(1, -1) * ((Pf @ Af) * Z))
    # Definir matrices de regularización
    O1 = -2 * np.ones((Nu, Nf))
    O2 = N * np.eye(Nu) - np.ones((Nu, Nu))

    # Gradiente
    if parallel:
        GradPK = Parallel(n_jobs=-1)(delayed(compute_grad)(X[:, k].copy()-V[:, k].copy(), Z[:, k].copy(), Au[:, k].copy(), D[k].copy(), Pu.copy(), onesL, Y[:, k].copy()) for k in range(K))
        GradP = np.sum(np.array(GradPK), axis=0)
    else:
        GradP = np.zeros((L, Nu))
        for k in range(K):
            GradP += compute_grad(X[:, k].copy()-V[:, k].copy(), Z[:, k].copy(), Au[:, k].copy(), D[k].copy(), Pu.copy(), onesL, Y[:, k].copy())
    
    GradP = GradP / K + rho * (Pu @ O2.T) / R + rho * (Pf @ O1.T) / (2 * R)

    # Computar el paso óptimo
    if parallel:
        numGg, denGg = zip(*Parallel(n_jobs=-1)(delayed(compute_num_den)(
            X[:, k].copy()-V[:, k].copy(), Z[:, k].copy(), Au[:, k].copy(), D[k].copy(), Pu, GradP.copy(), onesL, Y[:, k].copy()) for k in range(K)))
        numG = sum(numGg) + rho * np.trace(GradP @ O2 @ Pu.T + Pu @ O2 @ GradP.T + GradP @ O1 @ Pf.T) / (2 * R)
        denG = sum(denGg) + rho * np.trace(GradP @ O2 @ GradP.T) / R
    else:
        numG = rho * np.trace(GradP @ O2 @ Pu.T + Pu @ O2 @ GradP.T + GradP @ O1 @ Pf.T) / (2 * R)
        denG = rho * np.trace(GradP @ O2 @ GradP.T) / R
        for k in range(K):
            numGk, denGk = compute_num_den(X[:, k].copy()-V[:, k].copy(), Z[:, k].copy(), Au[:, k].copy(), D[k].copy(), Pu.copy(), GradP.copy(), onesL.copy(), Y[:, k].copy())
            numG += numGk
            denG += denGk

    alpha = max(0, numG / denG)

    # Actualización
    P_est = Pu - alpha * GradP
    P_est[P_est < 0] = 0
    P_est[np.isnan(P_est)] = 0
    P_est[np.isinf(P_est)] = 0
    Pc = np.hstack((Pf, P_est))
    P = Pc / np.sum(Pc, axis=0)

    return P


def compute_grad(sk, zk, ak, dk, Pu, onesL, yk):
    """
    Computar el gradiente para un solo `k`.
    """
    byk = np.dot(yk.T, yk)
    Mk = np.diag((1 - dk) * onesL + dk * zk)
    GradPK = -np.dot(Mk.T, sk[:, np.newaxis] * ak[np.newaxis, :]) / byk + (Mk.T @ Mk) @ Pu @ (ak[:, np.newaxis] @ ak[np.newaxis, :]) / byk
    return GradPK


def compute_num_den(sk, zk, ak, dk, Pu, GradP, onesL, yk):
    """
    Computar numerador y denominador para un solo `k`.
    """
    byk = np.dot(yk.T, yk)
    Mk = np.diag((1 - dk) * onesL + dk * zk)
    T1 = Mk @ GradP @ ak
    numGk = np.dot(T1.T, np.dot(Mk, Pu @ ak - sk)) / byk
    denGk = np.dot(T1.T, T1) / byk
    return numGk, denGk

def sparsenoise(Z, Y, P, A, D, lm):
    """
    Estimación del Componente de Ruido Disperso.

    Args:
        Z (numpy.ndarray): Matriz de mediciones originales.
        Y (numpy.ndarray): Matriz de mediciones normalizadas.
        P (numpy.ndarray): Matriz de end-members.
        A (numpy.ndarray): Matriz de abundancias.
        D (numpy.ndarray): Matriz de interacciones no lineales.
        lm (float): Peso para la estimación del ruido disperso.

    Returns:
        numpy.ndarray: Matriz de ruido disperso.
    """
    # Inicialización de la matriz de ruido disperso
    V = np.zeros_like(Y)
    # Computar el término E
    PA = np.dot(P, A)  # Producto matriz-matriz
    E = Y - ((1 - D).reshape(1, -1) * PA) - (D.reshape(1, -1) * (PA * Z))

    # Computar Ye
    Ye = np.sum(Y**2, axis=0, keepdims=True)  # Suma de cuadrados de cada columna

    # Estimación del ruido disperso
    V = np.sign(E) * np.maximum(0, np.abs(E) - lm * Ye)
    V = np.maximum(0, V)

    return V

def nesseae(Yo=[], N=2, parameters=[], Po=[]):

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
    Dm=np.zeros((K,1));

    if display==1:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
            print('Hybrid NEBEAE with Multi-Linear Unmixing & Sparse Noise');
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
        Am = abundance(Y,Ym,P,Dm,Vm,Lambda,parallel);
        Dm = probanonlinear(Y, Ym, P, Am, Vm)
        
        Pp = P.copy()
        P = hybrid_endmember(Y, Ym, Pp, Am, Dm, Vm, Nf, rho,parallel)
            
        Vm = sparsenoise(Y, Ym, P, Am, Dm, lm)
        Jp = J
        residual = Ym - ((1 - Dm)[np.newaxis, :] * (P @ Am)) - (Dm[np.newaxis, :] * ((P @ Am) * Y)) - Vm
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
        D = Dm.copy();
        V = Vm.copy();
    else:
        Ins = np.sort(list(set(I) - set(Is)))
        Ins = np.array(Ins, dtype=int)
        J = 1e5
        Jp = 1e6
        Dms = np.zeros(len(Ins))
        iter = 1
        Ymos = np.c_[Ymo[:, Ins]]
        Yos = np.c_[Yo[:, Ins]]
        Vms = np.zeros_like(Ymos)
        while ((Jp-J)/Jp) >= epsilon and iter<=maxiter:
            Ams = abundance(Yos, Ymos, P, Dms, Vms, Lambda, parallel)
            Dms = probanonlinear(Yos, Ymos, P, Ams, Vms, parallel)
            Vms = sparsenoise(Yos, Ymos, P, Ams, Dms, lm)
            Jp = J
            residual = Ymos - np.multiply((1 - Dms).reshape(1, -1), P @ Ams) - \
                       np.multiply(Dms.reshape(1, -1), np.multiply(P @ Ams, Yos)) - Vms
            J = np.linalg.norm(residual, 'fro')
            iter += 1
        
        A = np.hstack((Am, Ams))
        D = np.hstack((Dm, Dms))
        V = np.hstack((Vm, Vms))
        II = np.hstack((Is, Ins))
    
        # Sort and reorder
        Index = np.argsort(II)  # Get the indices for sorting
        A = A[:, Index]
        D = D[Index]
        V = V[:, Index]
    
    S=mYmo
    AA = A * mYmo
    V = V * mYmo
    
    Yh = (np.expand_dims(1-D, axis=0) * (P @ AA) + np.expand_dims(D, axis=0) * ((P @ AA) * Yo) +V)   
    Ds=probanonlinear(Yo,Yo,P,A,V,parallel);

    #return P, Am, Dm, Pp
    return P,A,Ds,S,Yh,V,Ji 

