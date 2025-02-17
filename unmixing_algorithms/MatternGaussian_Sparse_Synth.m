function [Y,P,A,V,D]=MatternGaussian_Sparse_Synth(SNR,density,ModelType)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
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
% y_k = \sum_{n=1}^N a_{k,n} p_n 
%         + \sum_{n=1}^{N} \sum_{m=1}^N \xi (a_{k,n}p_n) \odot (a_{k,m}p_m) + v_k
%
% 4) Multilinear Mixing Model (MMM) --> Heylen and Scheunders (2016)
%
%  y_k = (1-P_k) \sum_{n=1}^N a_{k,n} p_n / (1-P_k \sum_{n=1}^N a_{k,n} p_n) + v_k
%
% INPUTS
% N --> Order of multi-exponential model \in [2,4]
% Npixels --> numbers of pixels in x & y axes
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
%
%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Synthetic Dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%



if SNR ~= 0,
   NoiseGaussian=1;
else
   NoiseGaussian=0;
end
if density ~= 0,
   NoiseSparse=1; 
else
    NoiseSparse=0;
end
if SNR ~= 0 || density ~= 0,
    NoiseMeasurement=1;        
else
     NoiseMeasurement=0;
end

Nsamp=64;
Npixels=Nsamp;
K=Npixels*Npixels;
rng("shuffle");



load('DatasetSynth/EndMembersHbHbO2FatWater.mat')
L=size(P,1);
P1=P(:,1)/max(max(P));
P2=P(:,2)/max(max(P));
P3=P(:,3)/max(max(P));
P4=P(:,4)/max(max(P));

gamma0=[0.5 0.3 0.25 0.5 0.6 0.2]; % (2) Mixing coefficients in GBM
xi0=0.3;                          % (3) Scaling coefficient in PNMM
%prob=ones(Npixels,Npixels)*0.2;                          % (4) Probability of nonlinear mixing
prob=randn(Npixels, Npixels)*0.1+0.3;
prob(prob>1)=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of Abundance Maps and Nonlinear Interactions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('DatasetSynth/AbundancesMatternGaussian64.mat')
a1=squeeze(A(:,:,1));
a2=squeeze(A(:,:,2));
a3=squeeze(A(:,:,3));
a4=squeeze(A(:,:,4));
N=4;
Yy=zeros(Nsamp,Nsamp,L);
Gg=zeros(Nsamp,Nsamp,L);
Dd=zeros(Nsamp,Nsamp);

for i=1:Nsamp
    for j=1:Nsamp
        if N==2
            if ModelType==1
                g=(a1(i,j)*P1).*(a2(i,j)*P2);
            elseif ModelType==2
                %gamma=rand(sum(1:(N-1)),1);
                g=(a1(i,j)*P1).*(a2(i,j)*P2)*gamma(1);
            elseif ModelType==3
                g=(a1(i,j)*P1 + a2(i,j)*P2).*(a1(i,j)*P1 + a2(i,j)*P2)*xi;
            else
                g=0;
            end
            y=a1(i,j)*P1 + a2(i,j)*P2;
        elseif N==3
            if ModelType==1
                 g=(a1(i,j)*P1).*(a2(i,j)*P2) + (a1(i,j)*P1).*(a3(i,j)*P3) + (a2(i,j)*P2).*(a3(i,j)*P3);
            elseif ModelType==2
                %gamma=rand(sum(1:(N-1)),1);
                gammaL=gamma0+randn(6)*0.1;
                 g=(a1(i,j)*P1).*(a2(i,j)*P2)*gammaL(1)+(a1(i,j)*P1).*(a3(i,j)*P3)*gammaL(2)+(a2(i,j)*P2).*(a3(i,j)*P3)*gammaL(3);
            elseif ModelType==3
                 xi=xi0+randn*0.01;
                 g=(a1(i,j)*P1 + a2(i,j)*P2 + a3(i,j)*P3).*(a1(i,j)*P1 + a2(i,j)*P2 + a3(i,j)*P3)*xi;
            else
                g=0;
            end
            y=a1(i,j)*P1 + a2(i,j)*P2 + a3(i,j)*P3; 
        elseif N==4
            if ModelType==1
                g1=(a1(i,j)*P1).*(a2(i,j)*P2) + (a1(i,j)*P1).*(a3(i,j)*P3) + (a1(i,j)*P1).*(a4(i,j)*P4);
                g=g1 + (a2(i,j)*P2).*(a3(i,j)*P3) + (a2(i,j)*P2).*(a4(i,j)*P4) + (a3(i,j)*P3).*(a4(i,j)*P4);
            elseif ModelType==2
                %gamma=rand(sum(1:(N-1)),1);
                g1=(a1(i,j)*P1).*(a2(i,j)*P2)*gamma(1) + (a1(i,j)*P1).*(a3(i,j)*P3)*gamma(2) + (a1(i,j)*P1).*(a4(i,j)*P4)*gamma(3);
                g=g1 + (a2(i,j)*P2).*(a3(i,j)*P3)*gamma(4) + (a2(i,j)*P2).*(a4(i,j)*P4)*gamma(5) + (a3(i,j)*P3).*(a4(i,j)*P4)*gamma(6);
            elseif ModelType==3
                g=(a1(i,j)*P1 + a2(i,j)*P2 + a3(i,j)*P3 + a4(i,j)*P4).*(a1(i,j)*P1 + a2(i,j)*P2 + a3(i,j)*P3 + a4(i,j)*P4)*xi;
            else
                g=0;
            end
            y=a1(i,j)*P1 + a2(i,j)*P2 + a3(i,j)*P3 + a4(i,j)*P4; 
        end
                      
        Gg(i,j,:)=g;


        if ModelType==4
            x=y./sum(y);
            y=((1-prob(i,j))*x)./(1-(prob(i,j)*x));
            Dd(i,j)=prob(i,j);
        else 
            Dd(i,j)=0;
        end
        Gg(i,j,:)=g;     
        Yy(i,j,:)=y+g;
   
    end
end

%Ym=max(reshape(Yy,K,L)',[],2);
Ym=mean(reshape(Yy,K,L)',2);

if NoiseMeasurement==1 && NoiseGaussian==1
    sigmay=sqrt((1/(L-1))*(Ym'*Ym)/(10^(SNR/10)));
    Yy=Yy+sigmay*randn(Nsamp,Nsamp,L);
end

V=zeros(L,K);
if NoiseMeasurement==1 && NoiseSparse==1
     Iperp=randperm(K,round(K/1));
     V(:,Iperp)=sprand(L,length(Iperp),density)*0.2*max(Yy(:))';    
end  

if N==2     
    P=[P1 P2];
    A=[reshape(a1,1,K);reshape(a2,1,K)];
elseif N==3
    P=[P1 P2 P3];
    A=[reshape(a1,1,K);reshape(a2,1,K); reshape(a3,1,K)];
elseif N==4
    P=[P1 P2 P3 P4];
    A=[reshape(a1,1,K);reshape(a2,1,K); reshape(a3,1,K); reshape(a4,1,K)];
end

D=reshape(Dd,K,1);
Y=reshape(Yy,K,L)'+V;




