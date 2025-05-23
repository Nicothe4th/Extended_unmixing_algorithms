%% Setup
clear all; 
addpath('Matlab');

iters = 50;
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
ModelType=0;
% disp('1: EBEAE')
% disp('2: EBEAE-TV')
disp('3: EBEAE-SN')
% disp('4: EBEAE-SNTV')
% disp('5: ESSEAE')

for i =1: iters
    disp([' iter num: ',num2str(i)])
    [Y,P0,A0,V,D] = MatternGaussian_Sparse_Synth(SNR,0,ModelType);
    
    tic
    [P,A,S,Yh,Ji]=EBEAE(Y,N,par);
    t=toc;  
    E_zm(1,i)= round (norm(Yh-Y,'fro')/norm(Y,'fro'), 4);
    E_am(1,i)= round (errorabundances(A0,A), 4);
    E_pm(1,i)= round (errorendmembers(P0,P), 4);
    E_sm(1,i)= round (errorSAM(P0,P),4);
    E_tm(1,i)=round (t,4);

    
    tic
    [Ptv,Atv,Wtv,Stv,Yhtv,Jitv]=EBEAETV(Y,N,partv);
    ttv=toc;  
    E_zm(2,i)=norm(Yhtv-Y,'fro')/norm(Y,'fro');
    E_am(2,i)=errorabundances(A0,Atv);
    E_pm(2,i)=errorendmembers(P0,Ptv);
    E_sm(2,i)=errorSAM(P0,Ptv);
    E_tm(2, i)=ttv;
    

    [Y,P0,A0,V,D] = MatternGaussian_Sparse_Synth(SNR,density,ModelType);
    
    tic
    [Psn,Asn,Ssn,Yhsn,Vsn,Jisn]=EBEAESN(Y,N,parsn);
    tsn=toc;  
    E_zm(3,i)=norm(Yhsn-Y,'fro')/norm(Y,'fro');
    E_am(3,i)=errorabundances(A0,Asn);
    E_pm(3,i)=errorendmembers(P0,Psn);
    E_sm(3,i)=errorSAM(P0,Psn);
    E_tm(3,i)=tsn;
    %EBEAE-SNTV
    
    tic
    [Psntv,Asntv,Wsntv,Ssntv,Yhsntv,Vsntv,Jisntv]=EBEAESNTV(Y,N,parsntv);
    tsntv=toc;  
    E_zm(4,i)=norm(Yhsntv-Y,'fro')/norm(Y,'fro');
    E_am(4,i)=errorabundances(A0,Asntv);
    E_pm(4,i)=errorendmembers(P0,Psntv);
    E_sm(4,i)=errorSAM(P0,Psntv);
    E_tm(4,i)=tsntv;
    
    rNs =  sort(randperm(4, 2));
    Pu=P0(:,[1,2]);
    tic;
    [Pss,Ass,Sss,Zhss,Vss,Jss]=ESSEAE(Y,N,parsn,Pu);
    tesseae=toc;
    E_zm(5,i)=norm(Zhss-Y,'fro')/norm(Y,'fro');
    E_am(5,i)=errorabundances(A0,Ass);
    E_pm(5,i)=errorendmembers(P0,Pss);
    E_sm(5,i)=errorSAM(P0,Pss);
    E_tm(5,i)=tesseae;
end
%%
% functions = {'ebeae', 'ebeaetv', 'ebeaesn', 'ebeaesntv', 'esseae'};
% errors_lmm_matlab = {E_zm, E_am, E_pm, E_sm, E_tm};
% titles = {'Reconstruction Error', 'Abundance Error', 'Endmember Error', 'SAM Error', 'Computation Time'};
% 
% figure;
% for i = 1:5
%     subplot(1, 5, i); % Crear subgráficos en una fila de 5
%     data = squeeze(errors_lmm_matlab{i}); % Extraer la matriz de datos
%     
%     % Boxplot con notches
%     boxplot(data', 'Notch', 'on', 'Labels', functions);
%     
%     title(titles{i});
%     xtickangle(30); % Rotar etiquetas del eje X
% end
% 
% sgtitle('Comparison of Errors Across Methods'); % Título general

%% 
save('errors_lmm_matlab.mat', 'E_zm', 'E_am', 'E_pm','E_sm','E_tm', '-v7')