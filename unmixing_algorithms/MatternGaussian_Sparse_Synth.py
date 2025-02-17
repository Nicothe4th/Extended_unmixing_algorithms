#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% [Y,P,A,V,D]=MatternGaussian_Sparse_Synth(SNR,PSNR,ModelType)
%
% Hb, HbO2, Fat and Water Absorbance End-members with linear or non-linear mixing models:
%
% 0) Linear Mixing Model (LMM)
%
% y_k = \sum_{n=1}^N a_{k,n} p_n + v_k         v_k \sim N(0,\sigma^2 I)
%
% 1) Fan Model (FM) --> Fan et al., 2009
%
% y_k = \sum_{n=1}^N a_{k,n} p_n 
%         + \sum_{n=1}^{N-1} \sum_{m=n+1}^N (a_{k,n}p_n) \odot (a_{k,m}p_m) + v_k
%
% 2) Generalized Bilinear Model (GBM) --> Halimi et al., 2011)
%
% y_k = \sum_{n=1}^N a_{k,n} p_n 
%         + \sum_{n=1}^{N-1} \sum_{m=1}^N \gamma_{n,m} (a_{k,n}p_n) \odot (a_{k,m}p_m) + v_k
%
% 3) Postnonlinear Mixing Model (PNMM) --> Altmann et al., 2012
%
% y_k = \sum_{n=1}^N a_{k,n} p_n  + \sum_{n=1}^{N} \sum_{m=1}^N \ xi (a_{k,n}p_n) \odot (a_{k,m}p_m) + v_k
  4) Multilinear Mixing Model (MMM) --> Heylen and Scheunders (2016)
%
%  y_k = (1-P_k) \sum_{n=1}^N a_{k,n} p_n / (1-P_k \sum_{n=1}^N a_{k,n} p_n) + v_k
%
% INPUTS
% N --> Order of multi-exponential model \in [2,4]
% SNR --> SNR of Gaussian noise (dB)
% density --> density of sparse noise matrix with uniformly distributed
%             random entried
% ModelType --> 0 (LMM-Default), 1 (FM), 2 (GBM), 3 (PNMM) and 4 (MMM)
%
% OUTPUTS
% Y --> matrix of measurements of size 186 x (Npixels*Npixels)
% P --> matrix of end-members 186 x N
% A --> matrix of abundances of N x (Npixels*Npixels)
% V --> matrix of sparse noise
% D --> matrix of nonlinear interaction level of size (Npixels*Npixels)
%
%
% May/2024
% DUCD
@author: jnmc

"""
import numpy as np
from scipy.io import loadmat
import scipy.sparse as sparse

def MatternGaussian_Sparse_Synth(SNR,density,ModelType): 
    data = loadmat('DatasetSynth/EndMembersHbHbO2FatWater.mat')
    
    P=data['P']
    wavelength=data['wavelength'].ravel()
    

    
    
    NoiseGaussian = int(SNR != 0)
    NoiseSparse = int(density != 0)
    NoiseMeasurement = int(SNR != 0 or density != 0)
    
    Nsamp=64
    Npixels = Nsamp
    K = Npixels*Npixels
    np.random.seed(None)
    
    L, N = P.shape
    P1 = (P[:, 0] / np.max(P)).reshape(-1, 1)
    P2 = (P[:, 1] / np.max(P)).reshape(-1, 1)
    P3 = (P[:, 2] / np.max(P)).reshape(-1, 1)
    P4 = (P[:, 3] / np.max(P)).reshape(-1, 1)
    
    
    # (2) Mixing coefficients in GBM
    gamma0 = np.array([0.5, 0.3, 0.25, 0.5, 0.6, 0.2])
    
    # (3) Scaling coefficient in PNMM
    xi0 = 0.3
    
    # Generate random probabilities with added noise
    Npixels = 100  # Replace with your desired value for Npixels
    prob = np.random.randn(Npixels, Npixels) * 0.1 + 0.3
    prob[prob > 1] = 1
    
    data = loadmat('DatasetSynth/AbundancesMatternGaussian64.mat')
    A = data['A'] 
    
    a1 = A[:, :, 0]
    a2 = A[:, :, 1]
    a3 = A[:, :, 2]
    a4 = A[:, :, 3]
    
    Yy = np.zeros((Nsamp, Nsamp, L))
    Gg = np.zeros((Nsamp, Nsamp, L))
    Dd = np.zeros((Nsamp, Nsamp))
    
    for i in range(Nsamp):
        for j in range(Nsamp):
            if N == 2:
                if ModelType == 1:
                    g = (a1[i, j] * P1) * (a2[i, j] * P2)
                elif ModelType == 2:
                    g = (a1[i, j] * P1) * (a2[i, j] * P2) * gamma0[0]
                elif ModelType == 3:
                    g = (a1[i, j] * P1 + a2[i, j] * P2) * (a1[i, j] * P1 + a2[i, j] * P2) * xi0
                else:
                    g = 0
                y = a1[i, j] * P1 + a2[i, j] * P2
    
            elif N == 3:
                if ModelType == 1:
                    g = (a1[i, j] * P1) * (a2[i, j] * P2) + \
                        (a1[i, j] * P1) * (a3[i, j] * P3) + \
                        (a2[i, j] * P2) * (a3[i, j] * P3)
                elif ModelType == 2:
                    gammaL = gamma0 + np.random.randn(6) * 0.1
                    g = (a1[i, j] * P1) * (a2[i, j] * P2) * gammaL[0] + \
                        (a1[i, j] * P1) * (a3[i, j] * P3) * gammaL[1] + \
                        (a2[i, j] * P2) * (a3[i, j] * P3) * gammaL[2]
                elif ModelType == 3:
                    xi = xi0 + np.random.randn() * 0.01
                    g = (a1[i, j] * P1 + a2[i, j] * P2 + a3[i, j] * P3) * \
                        (a1[i, j] * P1 + a2[i, j] * P2 + a3[i, j] * P3) * xi
                else:
                    g = 0
                y = a1[i, j] * P1 + a2[i, j] * P2 + a3[i, j] * P3
    
            elif N == 4:
                if ModelType == 1:
                    g1 = (a1[i, j] * P1) * (a2[i, j] * P2) + \
                         (a1[i, j] * P1) * (a3[i, j] * P3) + \
                         (a1[i, j] * P1) * (a4[i, j] * P4)
                    g = g1 + (a2[i, j] * P2) * (a3[i, j] * P3) + \
                        (a2[i, j] * P2) * (a4[i, j] * P4) + \
                        (a3[i, j] * P3) * (a4[i, j] * P4)
                elif ModelType == 2:
                    g1 = (a1[i, j] * P1) * (a2[i, j] * P2) * gamma0[0] + \
                         (a1[i, j] * P1) * (a3[i, j] * P3) * gamma0[1] + \
                         (a1[i, j] * P1) * (a4[i, j] * P4) * gamma0[2]
                    g = g1 + (a2[i, j] * P2) * (a3[i, j] * P3) * gamma0[3] + \
                        (a2[i, j] * P2) * (a4[i, j] * P4) * gamma0[4] + \
                        (a3[i, j] * P3) * (a4[i, j] * P4) * gamma0[5]
                elif ModelType == 3:
                    g = (a1[i, j] * P1 + a2[i, j] * P2 + a3[i, j] * P3 + a4[i, j] * P4) * \
                        (a1[i, j] * P1 + a2[i, j] * P2 + a3[i, j] * P3 + a4[i, j] * P4) * xi
                else:
                    g = 0
                y = a1[i, j] * P1 + a2[i, j] * P2 + a3[i, j] * P3 + a4[i, j] * P4
    
            Gg[i, j, :] = g
    
            if ModelType == 4:
                x = y / np.sum(y, axis=0, keepdims=True)
                y = ((1 - prob[i, j]) * x) / (1 - (prob[i, j] * x))
                Dd[i, j] = prob[i, j]
            else:
                Dd[i, j] = 0
    
            Gg[i, j, :] = g
            Yy[i, j, :] = y.ravel() + g
    
    Ym = np.mean(Yy.reshape(K, L, order='F').T, axis=1).reshape(-1, 1)  
    
    if NoiseMeasurement == 1 and NoiseGaussian ==1:
        sigmay = np.sqrt( (1/(L-1)) * (Ym.T@Ym)/(10**(SNR/10)))[0,0]
        Yy = Yy + sigmay * np.random.randn(Nsamp, Nsamp, L)
    
    V = np.zeros((L,K))
    if NoiseMeasurement == 1 and NoiseSparse == 1:
        Iperp = np.random.permutation(K)[:int(round(K / 1))]
        sparse_part = sparse.random(L, len(Iperp), density=density, format="csc", data_rvs=lambda size: np.random.uniform(0, 1, size))
        V[:, Iperp] = sparse_part.toarray() * 0.2 * np.max(Yy)
        
    if N == 2:
        P = np.concatenate((P1, P2),axis=1)
        A = np.stack((a1.reshape(1, K, order='F').ravel(), a2.reshape(1, K, order='F').ravel()))
    elif N == 3:
        P = np.concatenate((P1, P2, P3),axis=1)
        A = np.stack((a1.reshape(1, K, order='F').ravel(), a2.reshape(1, K, order='F').ravel(), a3.reshape(1, K, order='F').ravel()))
    elif N == 4:
        P = np.concatenate((P1, P2, P3, P4),axis=1)
        A = np.stack((a1.reshape(1, K, order='F').ravel(), a2.reshape(1, K, order='F').ravel(), a3.reshape(1, K, order='F').ravel(), a4.reshape(1, K, order='F').ravel()))
        
    D = Dd.reshape(K,1,order='F').ravel()
    Y = Yy.reshape(K,L,order='F').T + V
    return Y,P,A,V,D