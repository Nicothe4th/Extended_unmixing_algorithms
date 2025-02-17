#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:46:44 2025

@author: jnmc
"""
from Python.EBEAE import ebeae
from Python.EBEAETV import ebeaetv
from Python.EBEAESN import ebeaesn
from Python.EBEAESNTV import ebeaesntv
from Python.ESSEAE import esseae
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

ModelType=0; 
SNR=35
density = 0.0075

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
parsntv =  [initcond, rho, 1e-4, lm,0.1, 10, nRow, nCol, epsilon, maxiter,  parallel, display_iter];

print('0: EBEAE')
print('1: EBEAETV')
print('2: EBEAESN')
print('3: EBEAESNTV')
print('4: ESSEAE')

for i in range (iters):
    print(f'iter num: {i}')
    Y,P0,A0,V,D = MG(SNR,0,ModelType)
    
    tic = time.perf_counter()
    P,A,S,Yh,Ji=ebeae(Y,N,par)
    t=time.perf_counter()-tic;
    E_zp[0,i]= np.round(np.linalg.norm(Yh - Y, 'fro') /np.linalg.norm(Y,'fro'),4)
    E_ap[0,i]= np.round(errorabundances(A0,A),4);
    E_pp[0,i]= np.round(errorendmembers(P0,P),4);
    E_sp[0,i]= np.round(errorSAM(P0,P),4);
    E_tp[0,i]= np.round(t,4);
    
    
    tic = time.perf_counter()
    Ptv,Atv,Wtv,Stv,Yhtv,Jitv=ebeaetv(Y,N,partv)
    ttv=time.perf_counter()-tic;
    E_zp[1,i]=np.linalg.norm(Yhtv - Y, 'fro') /np.linalg.norm(Y,'fro')
    E_ap[1,i]=errorabundances(A0,Atv);
    E_pp[1,i]=errorendmembers(P0,Ptv);
    E_sp[1,i]=errorSAM(P0,Ptv);
    E_tp[1,i]=ttv;
    
    Y,P0,A0,V,D = MG(SNR,density,ModelType)
    
    tic = time.perf_counter()
    Psn,Asn,Ssn,Yhsn,Vsn,Jisn=ebeaesn(Y,N,parsn)
    tsn=time.perf_counter()-tic;
    E_zp[2,i]=np.linalg.norm(Yhsn - Y, 'fro') /np.linalg.norm(Y,'fro')
    E_ap[2,i]=errorabundances(A0,Asn);
    E_pp[2,i]=errorendmembers(P0,Psn);
    E_sp[2,i]=errorSAM(P0,Psn);
    E_tp[2,i]=tsn;
    
    
    tic = time.perf_counter()
    Psntv,Asntv,Wsntv,Ssntv,Yhsntv,Vsntv,Jisntv=ebeaesntv(Y,N,parsntv)
    tsntv=time.perf_counter()-tic;
    E_zp[3,i]=np.linalg.norm(Yhsntv - Y, 'fro') /np.linalg.norm(Y,'fro')
    E_ap[3,i]=errorabundances(A0,Asntv);
    E_pp[3,i]=errorendmembers(P0,Psntv);
    E_sp[3,i]=errorSAM(P0,Psntv);
    E_tp[3,i]=tsntv;
    
    rNs = np.sort(np.random.choice(4, 2, replace=False))
    Pu = P0[:,0:2].copy()
    
    tic = time.perf_counter()   
    Pss,Ass,Sss,Yhss,Vss,Jiss=esseae(Y,N,parsn,Pu)
    tss=time.perf_counter()-tic;
    E_zp[4,i]=np.linalg.norm(Yhss - Y, 'fro') /np.linalg.norm(Y,'fro')
    E_ap[4,i]=errorabundances(A0,Ass);
    E_pp[4,i]=errorendmembers(P0,Pss);
    E_sp[4,i]=errorSAM(P0,Pss);
    E_tp[4,i]=tss;
errors_lmm_python = [E_zp, E_ap, E_pp, E_sp, E_tp]
np.save('errors_lmm_python.npy', errors_lmm_python)