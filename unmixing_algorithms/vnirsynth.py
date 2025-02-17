#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:10:20 2023

[Y,P,A,G,u,t]=VNIRsynthNLM(N,Npixels,SNR,PSNR,ModelType)
INPUTS
Npixels --> numbers of pixels in x & y axes
ModelType --> 0 (LMM-Default), 1 (FM), 2 (GBM), 3 (PNMM) and 4 (MMM)

OUTPUTS
 P --> matrix of end-members 186 x N
 A --> matrix of abundances of N x (Npixels*Npixels)
 Yo --> matrix of measurements
 Yo = P*A+G

"""
import numpy as np 
import scipy.io as sio

def VNIRsynthNLM(Nsamp, SNR, PSNR, ModelType):
    NoiseGaussian = int(SNR != 0)
    NoiseShot = int(PSNR != 0)
    NoiseMeasurement = int(SNR != 0 or PSNR != 0)
    x=np.array(range(Nsamp))
    y=np.array(range(Nsamp))
    xx,yy = np.meshgrid(x,y);
    K=Nsamp*Nsamp;
    data = sio.loadmat('EndMembersVNIR.mat')
    P = data['P']
    L , _ = P.shape
    ag=np.zeros((Nsamp))
    
    aa1 = 7 * np.exp(-0.005 * (xx - Nsamp/2)**2 - 0.005 * (yy - Nsamp/2)**2) + 0.5
    aa2 = 2.5 * np.exp(-0.001 * (xx - Nsamp)**2 - 0.001 * yy**2) + 2.5 * np.exp(-0.0001 * xx**2 - 0.001 * (yy - Nsamp)**2)
    aa3 = 3.5 * np.exp(-0.001 * xx**2 - 0.0001 * (yy - Nsamp)**2) + 2.5 * np.exp(-0.0001 * (xx - Nsamp)**2 - 0.001 * (yy - Nsamp)**2)
    
    
    P1 = P[:,1]
    P2 = P[:,2]
    P3 = P[:,3]
    

    ag = aa1+aa2+aa3
    
    P1 = P1/sum(P1)
    P2 = P2/sum(P2)
    P3 = P3/sum(P3)
    
    a1 = aa1/ag
    a2 = aa2/ag
    a3 =aa3/ag
    
    Po = np.array([P1, P2, P3])
    
    
    Yy = np.zeros((Nsamp,Nsamp,L));
    Gg = np.zeros((Nsamp,Nsamp,L));
    g = np.zeros((L,))
    y = np.zeros((L,))
    
    for i in range(Nsamp):
        for j in range(Nsamp):
            if ModelType==1:
                g=(a1[i,j]*P1)*(a2[i,j]*P2) + (a1[i,j]*P1)*(a3[i,j]*P3) + (a2[i,j]*P2)*(a3[i,j]*P3)
            elif ModelType==2:
                gamma=np.random.rand(3,1)
                g=(a1[i,j]*P1)*(a2[i,j]*P2)*gamma[0] + (a1[i,j]*P1)*(a3[i,j]*P3)*gamma[1] + (a2[i,j]*P2)*(a3[i,j]*P3)*gamma[2]
            elif ModelType==3:
                xi=np.random.uniform(-0.3, 0.3)
                g=(a1[i,j]*P1 + a2[i,j]*P2 + a3[i,j]*P3)*(a1[i,j]*P1 + a2[i,j]*P2 + a3[i,j]*P3)*xi
            else:
                g=0
            
            y=a1[i,j]*P1 + a2[i,j]*P2 + a3[i,j]*P3
            
            
            
            Gg[i, j, :] = g
            Yy[i, j, :] = y + g
            if ModelType==4:
                d=0.1+0.2*np.random.randn()
                Gg[i, j, :]=d
                sy=np.sum(y)
                x=y/sy
                Yy[i,j,:]=sy*((1-d)*x)/(1-d*x)
    
                
    Ym = np.mean(Yy.reshape((L, K)), axis=1)
    Pm = np.sum(np.abs(Yy) ** 2) / np.size(Yy)
    if NoiseMeasurement==1 and NoiseGaussian==1:
        #print('ruido Gaussiano')
        noisePowerdB = -SNR + 10 * np.log10(Pm)
        noisePower = 10 ** (noisePowerdB / 10)
        noise = np.random.normal(0, np.sqrt(noisePower), Yy.shape)
        
        #sigmay=np.sqrt((1/(L-1))*(Ym.T@Ym)/(10**(SNR/10)))
        #Yy=Yy+sigmay*np.random.randn(Nsamp,Nsamp,L)
        Yy = Yy + noise
    if NoiseMeasurement==1 and NoiseShot==1:
        #print('ruido Shot')
        sigmay=np.sqrt(np.max(Ym)**2 /(10**(PSNR/10)))
        shotNoise=(sigmay*np.tile(Ym,(1,Nsamp*Nsamp)).reshape((L,Nsamp*Nsamp)).T * np.random.randn(Nsamp*Nsamp,L)).reshape((Nsamp,Nsamp,L))
        Yy=Yy+shotNoise
    
    
    
    Go = np.reshape(Gg, (K, L)).T
    if ModelType==4:
        Go=Go[0,:]
        
    Yo = np.reshape(Yy, (K, L)).T
    Po = np.array([P1, P2, P3]).T
    A = np.squeeze(np.array([np.reshape(a1, (1, K)), np.reshape(a2, (1, K)), np.reshape(a3, (1, K))]))
    return Yo, Po, A, Go

