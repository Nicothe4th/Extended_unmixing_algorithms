%% Setup
clear all; 
addpath('Matlab');

iters =50;
E_zm = zeros(5,iters);
E_am = zeros(5,iters);
E_pm = zeros(5,iters);
E_sm = zeros(5,iters);
E_tm = zeros(5,iters);


N=4;                % Number of End-members
Nsamples=64;
nCol=Nsamples;
nRow=Nsamples;


%%
initcond=6;             % Initial condition of end-members matrix: 6 (VCA) and 8 (SISAL).
rho=0.1;               % Similarity weight in end-members estimation
lambda=0.1;             % Etropy weight for abundance estimation
epsilon=1e-3;
maxiter=20;
parallel=1;
downsampling=0.0;       % Downsampling in end-members estimation
display_iter=0;            % Display partial performance in BEAE
lm=0.1;
par = [initcond,rho,lambda, epsilon,maxiter,downsampling,parallel,display_iter];
parsn=[initcond,rho,lambda, lm, epsilon,maxiter,downsampling,parallel,display_iter];
partv =  [initcond, rho, 1e-4, 0.1, 10, nRow, nCol, epsilon, maxiter,  parallel, display_iter];
parsntv =  [initcond, rho, 1e-4, lm,0.1, 10, nRow, nCol, epsilon, maxiter,  parallel, display_iter];

%% First Part Only Gaussian Noise

SNR=35;
density = 0.0075;
ModelType=4;
disp('1: NEBEAE')
disp('2: NEBEAE-TV')
disp('3: NEBEAE-SN')
disp('4: NEBEAE-SNTV')
disp('5: NESSEAE')

for i =1: iters
    disp(['start iter num =', num2str(i)])
    [Y,P0,A0,V0,D0] = MatternGaussian_Sparse_Synth(SNR,0,ModelType);
    tic
    %disp('1: NEBEAE')
    [P,A,D,S,Yh,Ji]=NEBEAE(Y,N,par);
    t=toc;  
    E_zm(1,i)=norm(Yh-Y,'fro')/norm(Y,'fro');
    E_am(1,i)=errorabundances(A0,A);
    E_pm(1,i)=errorendmembers(P0,P);
    E_sm(1,i)=errorSAM(P0,P);
    E_tm(1,i)=t;

    %disp('2: NEBEAE-TV')
    tic;
    [Ptv,Atv,Wtv, Dtv,Stv,Yhtv,Jitv]=NEBEAETV(Y,N,partv);
    ttv=toc;  
    E_zm(2,i)=norm(Yhtv-Y,'fro')/norm(Y,'fro');
    E_am(2,i)=errorabundances(A0,Atv);
    E_pm(2,i)=errorendmembers(P0,Ptv);
    E_sm(2,i)=errorSAM(P0,Ptv);
    E_tm(2, i)=ttv;

    [Y,P0,A0,V,D] = MatternGaussian_Sparse_Synth(SNR,density,ModelType);
    %disp('3: NEBEAE-SN')
    tic
    [Psn,Asn,Dsn,Ssn,Yhsn,Vsn,Jisn]=NEBEAESN(Y,N,parsn);
    tsn=toc;  
    E_zm(3,i)=norm(Yhsn-Y,'fro')/norm(Y,'fro');
    E_am(3,i)=errorabundances(A0,Asn);
    E_pm(3,i)=errorendmembers(P0,Psn);
    E_sm(3,i)=errorSAM(P0,Psn);
    E_tm(3,i)=tsn;
   
     %disp('4: NEBEAE-SNTV')
    tic
    [Psntv,Asntv,Wsntv,Dsntv,Ssntv,Yhsntv,Vsntv,Jisntv]=NEBEAESNTV(Y,N,parsntv);
    tsntv=toc;  
    E_zm(4,i)=norm(Yhsntv-Y,'fro')/norm(Y,'fro');
    E_am(4,i)=errorabundances(A0,Asntv);
    E_pm(4,i)=errorendmembers(P0,Psntv);
    E_sm(4,i)=errorSAM(P0,Psntv);
    E_tm(4,i)=tsntv;


    rNs =  sort(randperm(4, 2));
    Pu=P0(:,[1,2]);
    tic;
    %disp('5: NESSEAE')
    [Pss,Ass,Dss,Sss,Zhss,Vss,Jss]=NESSEAE(Y,N,parsn,Pu);
    tss=toc;
    E_zm(5,i)=norm(Zhss-Y,'fro')/norm(Y,'fro');
    E_am(5,i)=errorabundances(A0,Ass);
    E_pm(5,i)=errorendmembers(P0,Pss);
    E_sm(5,i)=errorSAM(P0,Pss);
    E_tm(5,i)=tss;
end


save('errors_mmm_matlab.mat', 'E_zm', 'E_am', 'E_pm','E_sm','E_tm', '-v7')