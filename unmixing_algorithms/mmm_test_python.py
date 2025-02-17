#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:48:18 2025

@author: jnmc
"""

from Python.NEBEAE import nebeae
from Python.NEBEAETV import nebeaetv
from Python.NEBEAESN import nebeaesn
from Python.NEBEAESNTV import nebeaesntv
from Python.NESSEAE import nesseae
from aux import errorabundances, errorendmembers, errorSAM
import time

import numpy as np
from MatternGaussian_Sparse_Synth import  MatternGaussian_Sparse_Synth as MG

iters = 50;
E_zp = np.zeros((5,iters));
E_ap = np.zeros((5,iters));
E_pp= np.zeros((5,iters));
E_sp = np.zeros((5,iters)); #SAM ERROR
E_tp = np.zeros((5,iters));

N=4;               
Nsamples=64;
nCol=Nsamples;
nRow=Nsamples;


initcond=6;             #% Initial condition of end-members matrix: 6 (VCA) and 8 (SISAL).
rho=0.1;               #% Similarity weight in end-members estimation
Lambda=0.1;             #% Etropy weight for abundance estimation
epsilon=1e-3;
maxiter=20;
parallel=1;
downsampling=0.0;       #Downsampling in end-members estimation
display_iter=0;            #Display partial performance in BEAE
lm=0.1;

par = [initcond,rho,Lambda, epsilon,maxiter,downsampling,parallel,display_iter];
parsn=[initcond,rho,Lambda, lm, epsilon,maxiter,downsampling,parallel,display_iter];
partv =  [initcond, rho, 1e-4, 0.1, 10, nRow, nCol, epsilon, maxiter,  parallel, display_iter];
parsntv =  [initcond, rho, 1e-4, lm, 0.1, 10, nRow, nCol, epsilon, maxiter,  parallel, display_iter];


ModelType=4; 
SNR=35
density = 0.0075
print('0: NEBEAE')
print('1: NEBEAE-TV')
print('2: NEBEAE-SN')
print('3: NEBEAE-SNTV')
print('4: NESSEAE')





for i in range (iters):
    print(f'start iter = {i}')
    Y,P0,A0,V,D = MG(SNR,0,ModelType)
    #print('0: NEBEAE')
    tic = time.perf_counter()
    P,A,Ds,S,Yh,Ji=nebeae(Y,N,par)
    t=time.perf_counter()-tic;
    E_zp[0,i]=np.linalg.norm(Yh - Y, 'fro') /np.linalg.norm(Y,'fro')
    E_ap[0,i]=errorabundances(A0,A);
    E_pp[0,i]=errorendmembers(P0,P);
    E_sp[0,i]=errorSAM(P0,P);
    E_tp[0,i]=t;
    
    ##print('1: NEBEAE-TV')
    tic = time.perf_counter()
    Ptv,Atv,Wtv,Dstv,Stv,Yhtv,Jitv=nebeaetv(Y,N,partv)
    ttv=time.perf_counter()-tic;
    E_zp[1,i]=np.linalg.norm(Yhtv - Y, 'fro') /np.linalg.norm(Y,'fro')
    E_ap[1,i]=errorabundances(A0,Atv);
    E_pp[1,i]=errorendmembers(P0,Ptv);
    E_sp[1,i]=errorSAM(P0,Ptv);
    E_tp[1,i]=ttv;
    
    Y,P0,A0,V,D = MG(SNR,density,ModelType)
    
    # #print('2: NEBEAESN')
    tic = time.perf_counter()
    Psn,Asn,Dssn,Ssn,Yhsn,Vsn,Jisn=nebeaesn(Y,N,parsn)
    tsn=time.perf_counter()-tic;
    E_zp[2,i]=np.linalg.norm(Yhsn - Y, 'fro') /np.linalg.norm(Y,'fro')
    E_ap[2,i]=errorabundances(A0,Asn);
    E_pp[2,i]=errorendmembers(P0,Psn);
    E_sp[2,i]=errorSAM(P0,Psn);
    E_tp[2,i]=tsn;
    
    
    # print('3: NEBEAE-SNTV')
    tic = time.perf_counter()
    Psntv,Asntv,Wsntv,Dsntv,Ssntv,Yhsntv,Vsntv,Jisntv=nebeaesntv(Y,N,parsntv)
    tsntv=time.perf_counter()-tic;
    E_zp[3,i]=np.linalg.norm(Yhsntv - Y, 'fro') /np.linalg.norm(Y,'fro')
    E_ap[3,i]=errorabundances(A0,Asntv);
    E_pp[3,i]=errorendmembers(P0,Psntv);
    E_sp[3,i]=errorSAM(P0,Psntv);
    E_tp[3,i]=tsntv;
    
    rNs = np.sort(np.random.choice(4, 2, replace=False))
    Pu = P0[:,0:2].copy()
    
    # print('4: NESSEAE')
    tic = time.perf_counter()   
    Pss,Ass,Dss,Sss,Yhss,Vss,Jiss=nesseae(Y,N,parsn,Pu)
    tss=time.perf_counter()-tic;
    E_zp[4,i]=np.linalg.norm(Yhss - Y, 'fro') /np.linalg.norm(Y,'fro')
    E_ap[4,i]=errorabundances(A0,Ass);
    E_pp[4,i]=errorendmembers(P0,Pss);
    E_sp[4,i]=errorSAM(P0,Pss);
    E_tp[4,i]=tss;
    
    
    
errors_mmm_python = [E_zp, E_ap, E_pp, E_sp, E_tp]
np.save('errors_mmm_python.npy', errors_mmm_python)