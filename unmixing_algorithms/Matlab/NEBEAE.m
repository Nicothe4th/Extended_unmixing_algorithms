function [P,A,Ds,S,Yh,Ji]=NEBEAE(Yo,N,parameters,Po,oae)

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
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default hyper-parameters of NEBEAE algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NUMERROR

initcond=6;
rho=0.1;
lambda=0;
epsilon=1e-3;
maxiter=20;
downsampling=0.5;
parallel=0;
display=0;
NUMERROR=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check concistency of input arguments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin~=5
    oae=0;
end
if nargin==0
    disp('The measurement matrix Y has to be used as argument!!');
    return;
elseif nargin==1
    N=2;
end
if nargin==3 || nargin==4 || nargin==5
    if length(parameters)~= 8
        disp('The length of parameters vector is not 8 !!');
        disp('Default values of hyper-parameters are used instead');
    else
        initcond=round(parameters(1));
        rho=parameters(2);
        lambda=parameters(3);
        epsilon=parameters(4);
        maxiter=parameters(5);
        downsampling=parameters(6);
        parallel=parameters(7);
        display=parameters(8);
        if initcond~=1 && initcond~=2 && initcond~=3 && initcond~=4 && initcond~=5 && initcond~=6 && initcond~=7 && initcond~=8
            disp('The initialization procedure of end-members matrix is 1 to 8!');
            disp('The default value is considered!');
            initcond=6;
        end
        if rho<0
            disp('The regularization weight rho cannot be negative');
            disp('The default value is considered!');
            rho=0.1;
        end
        if lambda<0 || lambda>=1
            disp('The entropy weight lambda is limited to [0,1)');
            disp('The default value is considered!');
            lambda=0;
        end
        if epsilon<0 || epsilon>0.5
            disp('The threshold epsilon cannot be negative or >0.5');
            disp('The default value is considered!');
            epsilon=1e-3;
        end
        if maxiter<0 && maxiter<100
            disp('The upper bound maxiter cannot be negative or >100');
            disp('The default value is considered!');
            maxiter=20;
        end
        if downsampling<0 && downsampling>1
            disp('The downsampling factor cannot be negative or >1');
            disp('The default value is considered!');
            downsampling=0.5;
        end
        if parallel~=0 && parallel~=1
            disp('The parallelization parameter is 0 or 1');
            disp('The default value is considered!');
            parallel=0;
        end
        if display~=0 && display~=1
            disp('The display parameter is 0 or 1');
            disp('The default value is considered!');
            display=0;
        end
    end
    if N<2
        disp('The order of the linear mixture model has to greater than 2!');
        disp('The default value N=2 is considered!');
        N=2;
    end
end
if nargin==4 || nargin==5
    if ~ismatrix(Po)
        disp('The initial end-members Po must be a matrix !!');
        disp('The initialization is considered by VCA from the input dataset');
        initcond=6;
    else
        if size(Po,1)==size(Yo,1) && size(Po,2)==N
            initcond=0;
        else
            disp('The size of Po must be LxN!!');
            disp('The initialization is considered by VCA from the input dataset');
            initcond=6;
        end
    end
end
if nargin==5
    if oae~=0 && oae~=1
        disp('The assignment of oae is incorrect!!');
        disp('The initial end-members Po will be improved iteratively from a selected sample');
        oae=0;
    elseif oae==1 && initcond~=0
        disp('The initial end-members Po is not defined properly!');
        disp('Po will be improved iteratively from a selected sample');
        oae=0;
    end
end
if nargin>6
    disp('The number of input arguments is 5 maximum');
    disp('Please check the help documentation');
    return;
end
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Random downsampling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~ismatrix(Yo)
    disp('The measurements matrix Y has to be a matrix');
    return;
end
[L,K]=size(Yo);
if L>K
    disp('Number of spatial measurements has to be larger than the number of time samples!');
    return;
end

I=1:K;
Kdown=round(K*(1-downsampling));
Is=randperm(K,Kdown);
Y=Yo(:,Is);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mYm=sum(Y,1);
mYmo=sum(Yo,1);
Ym=Y./repmat(mYm,[L 1]);
Ymo=Yo./repmat(mYmo,[L 1]);
NYm=norm(Ym,'fro');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Selection of Initial End-members Matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if initcond==1 || initcond==2
    if initcond==1
        Po=zeros(L,N);
        index=1;
        pmax=mean(Yo,2);
        Yt=Yo;
        Po(:,index)=pmax;
    elseif initcond==2
        index=1;
        Y1m=sum(abs(Yo),1);
        [~,Imax]=max(Y1m);
        [~,Imin]=min(Y1m);
        pmax=Yo(:,Imax);
        pmin=Yo(:,Imin);
        Kdown=size(Yo,2);
        II=1:Kdown;
        Yt=Yo(:,setdiff(II,[Imax Imin]));
        Po(:,index)=pmax;
        index=index+1;
        Po(:,index)=pmin;
    end
    while index<N
        ymax=zeros(1,index);
        Imax=zeros(1,index);
        for i=1:index
            e1m=sum(Yt.*repmat(Po(:,i),1,size(Yt,2)),1)./sqrt(sum(Yt.^2,1))./sqrt(sum(Po(:,i).^2,1));
            [ymax(i),Imax(i)]=min(abs(e1m));
        end
        [~,Immax]=min(ymax);
        IImax=Imax(Immax);
        pmax=Yt(:,IImax);
        index=index+1;
        Po(:,index)=pmax;
        II=1:size(Yt,2);
        Yt=Yt(:,setdiff(II,IImax));
    end
elseif initcond==3
    [~,~,VV]=svd(Ym',0);
     W=VV(:,1:N);
     Po=W.*repmat(sign(W'*ones(L,1))',L,1); 
elseif initcond==4
    Yom=mean(Ym,2);
    Yon = Ym - repmat(Yom,1,N);
    [~,S,VV]=svd(Yon',0);
    Yo_w= pinv(sqrtm(S))*VV'*Ym; 
    [V,~,~] = svd((repmat(sum(Yo_w.*Yo_w,1),M,1).*Yo_w)*Yo_w');
    W=VV*sqrtm(S)*V(1:N,:)'; 
    Po=W.*repmat(sign(W'*ones(L,1))',L,1);
elseif initcond==5
    Po=NFINDR(Ym,N);
elseif initcond==6
    Po=VCA(Ym,N);
elseif initcond==7
    Po=SVMAX(Ym,N);
elseif initcond==8
    Po=SISAL(Ym,N);
elseif initcond~=0
    disp('The selection of initial condition is incorrect !');
    disp('VCA is adopted by default');
    Po=VCA(Ym,N);
end      


Po(Po<0)=0;
Po(isnan(Po))=0;
Po(isinf(Po))=0;
mPo=sum(Po,1);
P=Po./repmat(mPo,[L 1]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Alternated Least Squares Procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

iter=1;
J=1e5;
Jp=1e6;
Ji=[];
Dm=zeros(K,1);

if display==1
        disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
        disp('NEBEAE with Multi-Linear Unmixing');
        disp(['Model Order =' num2str(N)]);
        if oae==1
            disp('Only the abundances and nonlinear interaction levels are estimated from Po');
        elseif oae==0 && initcond==0
            disp('The end-members matrix is initialized externally by matrix Po');
        elseif oae==0 && initcond==1
            disp('Po is constructed based on the maximum cosine difference from mean measurement'); 
        elseif oae==0 && initcond==2
            disp('Po is constructed based on the maximum and minimum energy, and largest difference from them');
        elseif oae==0 && initcond==3
            disp('Po is constructed based on the PCA selection + Rectified Linear Unit');
        elseif oae==0 && initcond==4
            disp('Po is constructed based on the ICA selection (FOBI) + Rectified Linear Unit');
        elseif oae==0 && initcond==5
            disp('Po is constructed based on N-FINDR endmembers estimation by Winter (1999)');
         elseif oae==0 && initcond==6
            disp('Po is constructed based on Vertex Component Analysis by Nascimento and Dias (2005)');
        elseif oae==0 && initcond==7
            disp('Po is constructed based on Simplex Volume Maximization (SVMAX) (Chan et al. 2011)');
        elseif oae==0 && initcond==8
            disp('Po is constructed based on Simplex identification via split augmented Lagrangian (SISAL) (Bioucas-Dias, 2009)');
        end
end

while (Jp-J)/Jp >= epsilon && iter < maxiter && oae==0 && NUMERROR==0
    
    Am = abundance(Y,Ym,P,Dm,lambda,parallel);
    Dm = probanonlinear(Y,Ym,P,Am,parallel);
    Pp = P;
   if NUMERROR==0
         P = endmember(Y,Ym,Pp,Am,Dm,rho,parallel); 
     end

    Jp=J;
    residual = Ym - (repmat((1-Dm)',[L,1]).*(P*Am)) - ( repmat(Dm',[L,1]).*((P*Am).*Y));
    J=norm(residual,'fro');
    Ji(iter)=(Jp-J)/Jp;
    if J > Jp
        P=Pp; 
        if display ==1
        disp(['Break Number of iteration =' num2str(iter)]);
        disp(['Break Percentage Estimation Error =' num2str(100*J/NYm) '%']);
        end
        break;

    end
    if display ==1
        disp(['Number of iteration =' num2str(iter)]);
        disp(['Percentage Estimation Error =' num2str(100*J/NYm) '%']);
    end
    iter=iter+1;
    
end

if NUMERROR==0
  
    if oae==1
        J=1e5;
        Jp=1e6;
        Ji=[];
        D=zeros(Kdown,1);
        iter=1;
        
        while (Jp-J)/Jp >= epsilon && iter < maxiter
            
            A=abundance(Yo,Ymo,P,D,lambda,parallel);
            D=probanonlinear(Yo,Ymo,P,A,parallel);
            
            Jp=J;
            J=norm(Ymo - repmat((1-D)',[L,1]).*(P*A) - repmat(D',[L,1]).*((P*A).*Yo),'fro');
            Ji(iter)=J;
            iter=iter+1;
            
        end
        if display==1
            disp(['Percentage Estimation Error =' num2str(100*J/NYm) '%']);
        end
    else
        Ins=setdiff(I,Is); 
        J=1e5;
        Jp=1e6;
        Dms=zeros(length(Ins),1);
        iter=1;
        Ymos=Ymo(:,Ins);
        Yos=Yo(:,Ins);
       
        while (Jp-J)/Jp >= epsilon && iter < maxiter
            
            Ams=abundance(Yos,Ymos,P,Dms,lambda,parallel);
            Dms=probanonlinear(Yos,Ymos,P,Ams,parallel);
            Jp=J;
            J=norm(Ymos - repmat((1-Dms)',[L,1]).*(P*Ams) - repmat(Dms',[L,1]).*((P*Ams).*Yos),'fro');
            iter=iter+1;
            
        end        
        
        A=[Am Ams];
        D=[Dm; Dms];
        II=[Is Ins];
        [~,Index]=sort(II);
        A=A(:,Index);
        D=D(Index);
    end
    ElapTime=toc;
    if display ==1
            disp(['Elapsep Time =' num2str(ElapTime)]);
    end
    S=mYmo';
    AA=(A.*repmat(mYmo,[N,1]));
    Yh=repmat((1-D)',[L,1]).*(P*AA) + repmat(D',[L,1]).*((P*AA).*Yo);
    Ds=probanonlinear(Yo,Yo,P,A,parallel);
else
    disp('Please revise the problem formulation, not reliable results');
    sound(0.1*sin(2*pi*(1:1000)/10))
    P=[];
    Ds=[];
    S=[];
    A=[];
    Yh=[];
end
end

%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%

function A = abundance(Z,Y,P,D,lambda,parallel)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% A = abundance(Z,Y,P,D,lambda,parallel)
%
% Estimation of Optimal Abundances in Nonlinear Mixture Model
%
% Input Arguments
% Z --> matrix of measurements
% Y --> matrix of normalized measurements
% P --> matrix of end-members
% D --> vector of probabilities of nonlinear mixing
% lambda -->  entropy weight in abundance estimation \in (0,1)
% parallel --> implementation in parallel of the estimation
%
% Output Argument
% A = abundances matrix 
%
% Daniel U. Campos-Delgado
% February/2021
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check arguments dimensions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NUMERROR

[L,K]=size(Y);
N=size(P,2);
A=zeros(N,K);
em=eye(N);

if size(P,1) ~= L
    disp('ERROR: the number of rows in Y and P does not match');
    NUMERROR=1;
    sound(0.1*sin(2*pi*(1:1000)/10))
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start Computation of Abundances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if parallel==1
    
    parfor k=1:K

        yk=Y(:,k);
        zk=Z(:,k);
        byk=yk'*yk;
        dk=D(k);
        deltakn=(1-dk)*ones(N,1)+dk*P'*zk;
        Pk=P.*((1-dk)*ones(L,N)+dk*zk*ones(1,N));
        bk=Pk'*yk;
        Go=Pk'*Pk;
        eGo=eig(Go);
        eGo(isnan(eGo))=1e6;
        eGo(isinf(eGo))=1e6;
        lmin=min(eGo);
        G=Go-em*lmin*lambda;
        Gi=inv(G)
  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute Optimal Unconstrained Solution
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        sigma=(deltakn'*Gi*bk-1)/(deltakn'*Gi*deltakn);
        ak = Gi*(bk-deltakn*sigma);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check for Negative Elements
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if(sum(ak>=0) ~=N)

            Iset = zeros(1,N);

            while(sum(ak<0) ~= 0)    

                Iset(ak<0) = 1;
                Ll = length(find(Iset));

                Q = N+1+Ll;
                Gamma = zeros(Q);
                Beta = zeros(Q,1);

                Gamma(1:N,1:N) = G;
                Gamma(1:N,N+1) = deltakn*byk;
                Gamma(N+1,1:N) = deltakn';

                cont = 0;
                for i = 1:N
                    if(Iset(i)~= 0)
                        cont = cont + 1;
                        ind = i; 
                        Gamma(ind,N+1+cont) = 1;
                        Gamma(N+1+cont,ind) = 1;   
                    end
                end

                Beta(1:N) = bk;
                Beta(N+1) = 1;
                delta = Gamma\Beta;
                ak = delta(1:N);
                ak(abs(ak)<1e-9) = 0;
            end    

       end

        A(:,k) = ak; 
    end
    
else
    
    for k=1:K

        yk=Y(:,k);
        zk=Z(:,k);
        byk=yk'*yk;
        dk=D(k);
        deltakn=(1-dk)*ones(N,1)+dk*P'*zk;
        Pk=P.*((1-dk)*ones(L,N)+dk*zk*ones(1,N));
        bk=Pk'*yk;
        Go=Pk'*Pk;
        eGo=eig(Go);
        eGo(isnan(eGo))=1e6;
        eGo(isinf(eGo))=1e6;
        lmin=min(eGo);
        G=Go-em*lmin*lambda;
        Gi=inv(G);
  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute Optimal Unconstrained Solution
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        sigma=(deltakn'*Gi*bk-1)/(deltakn'*Gi*deltakn);
        ak = Gi*(bk-deltakn*sigma);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check for Negative Elements
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if(sum(ak>=0) ~=N)

            Iset = zeros(1,N);

            while(sum(ak<0) ~= 0)    

                Iset(ak<0) = 1;
                Ll = length(find(Iset));

                Q = N+1+Ll;
                Gamma = zeros(Q);
                Beta = zeros(Q,1);

                Gamma(1:N,1:N) = G;
                Gamma(1:N,N+1) = deltakn*byk;
                Gamma(N+1,1:N) = deltakn';

                cont = 0;
                for i = 1:N
                    if(Iset(i)~= 0)
                        cont = cont + 1;
                        ind = i; 
                        Gamma(ind,N+1+cont) = 1;
                        Gamma(N+1+cont,ind) = 1;   
                    end
                end

                Beta(1:N) = bk;
                Beta(N+1) = 1;
                delta = Gamma\Beta;
                ak = delta(1:N);
                ak(abs(ak)<1e-9) = 0;
            end    
        end

        A(:,k) = ak; 
    end
    
end
end

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

function D=probanonlinear(Z,Y,P,A,parallel)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  P = probanonlinear(Z,Y,P,A,parallel)
%
% Estimation of Probability of Nonlinear Mixtures 
%
% Input Arguments
% Z --> matrix of measurements
% Y -> matrix of normalized measurements
% P --> matrix of end-members
% A -->  matrix of abundances
% parallel = implementation in parallel of the estimation
% 
% Output Arguments
% D = Vector of probabilities of Nonlinear Mixtures
%
% Daniel U. Campos-Delgado
% February/2021
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K=size(Y,2);
D=zeros(K,1);
G=sum(Z);

if parallel==1
    
    parfor k=1:K

        yk=Y(:,k);
        zk=Z(:,k);
        ak=A(:,k);
        ek=P*ak;        
        T1=ek-yk;
        T2=ek-ek.*zk;
        dk=min([1 T1'*T2/(T2'*T2)]);
        D(k)=dk;
        
    end
    
else
    
    for k=1:K

        yk=Y(:,k);
        zk=Z(:,k);
        ak=A(:,k);
        ek=P*ak;   
        T1=ek-yk;
        T2=ek-ek.*zk;
        dk=min([1 T1'*T2/(T2'*T2)]);
        D(k)=dk;
        
    end
    
end
end

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Separate Gradient Computation (matches Python function)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function GradPK = compute_grad_pk(yk, zk, ak, dk, Po, L)
    onesL = ones(L,1);
    Mk = diag((1 - dk) * onesL + dk * zk);
    byk = yk' * yk;
    GradPK = - Mk' * (yk * ak') / byk + Mk' * Mk * Po * (ak * ak') / byk;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Separate numG and denG Computation (matches Python function)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [numG, denG] = compute_num_den(yk, zk, ak, dk, Po, GradP, L)
    onesL = ones(L,1);
    Mk = diag((1 - dk) * onesL + dk * zk);
    byk = yk' * yk;
    T1 = Mk * GradP * ak;
    numG = T1' * Mk * (Po * ak - yk) / byk;
    denG = T1' * T1 / byk;
end
function P = endmember(Z,Y,Po,A,D,rho,parallel)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  P = endmember(Z,Y,Po,A,D,rho,parallel)
%
% Estimation of Optimal End-members in Multi-Linear Mixture Model
%
% Input Arguments
% Z --> matrix of measurements
% Y -> matrix of normalized measurements
% Po --> Initial matrix of end-members
% A -->  matrix of abundances
% D --> vector of nonlinear interaction levels 
% rho = Weighting factor of regularization term
% parallel --> parallel computations (0--> NO, 1 --> YES)
% 
% Output Arguments
% P --> Updated matrix of end-members
%
% Adjusted for numerical consistency with Python implementation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NUMERROR

[N,K] = size(A);
L = size(Y,1);
R = sum(N - (1:(N-1)));
em = eye(N);
onesL = ones(L,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Gradient of Cost Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if parallel == 1 
    GradPK = zeros(L,N,K);
    parfor k = 1:K
        GradPK(:,:,k) = compute_grad_pk(Y(:,k), Z(:,k), A(:,k), D(k), Po, L);
    end
    GradP = sum(GradPK, 3);
else
    GradP = zeros(L,N);
    for k = 1:K
        GradP = GradP + compute_grad_pk(Y(:,k), Z(:,k), A(:,k), D(k), Po, L);
    end
end

O = N * em - ones(N,N);  
GradP = GradP / K + rho * (Po * O) / R;  % Ensure Python-like division

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Optimal Step in Update Rule
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numG = 0;
denG = 0;

if parallel == 1
    numGg = zeros(K,1);
    denGg = zeros(K,1);
    parfor k = 1:K
        [numGg(k), denGg(k)] = compute_num_den(Y(:,k), Z(:,k), A(:,k), D(k), Po, GradP, L);
    end
    numG = sum(numGg) + rho * trace(GradP * O * Po' + Po * O * GradP') / (2 * R);
    denG = sum(denGg) + rho * trace(GradP * O * GradP') / R;
else
    for k = 1:K
        [num_k, den_k] = compute_num_den(Y(:,k), Z(:,k), A(:,k), D(k), Po, GradP, L);
        numG = numG + num_k;
        denG = denG + den_k;
    end
    numG = numG + rho * trace(GradP * O * Po' + Po * O * GradP') / (2 * R);
    denG = denG + rho * trace(GradP * O * GradP') / R;
end

% Handle numerical issues in step size calculation
if denG == 0
    alpha = 0;
else
    alpha = max(0, numG / denG);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the Steepest Descent Update of End-members Matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P_est = Po - alpha * GradP;
P_est(P_est < 0) = 0; 
P_est(isnan(P_est)) = 0;
P_est(isinf(P_est)) = 0;
P = P_est ./ sum(P_est, 1); % Ensure consistent normalization

end





%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

function Po = NFINDR(Y,N)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [P,indices] = NFINDR(Y,N)
%
% N-FINDR endmembers estimation in multi/hyperspectral dataset
%
% Inputs
%   Y --> Multi/hyperspectral dataset as 2D matrix (L x K).
%   N --> Number of endmembers to find.
%
% Outputs
%   P --> Matrix of endmembers (L x N).
%   indices --> Indicies of pure pixels in Y
%
% Bibliographical references:
% [1] Winter, M. E., «N-FINDR: an algorithm for fast autonomous spectral 
%     end-member determination in hyperspectral data», presented at the 
%     Imaging Spectrometry V, Denver, CO, USA, 1999, vol. 3753, págs. 266-275.
%
% DUCD February/2021
% IICO-FC-UASLP
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% data size
[L,K] = size(Y);

%% Dimensionality reduction by PCA
U = pca(Y,N);
Yr= U.'*Y;

%% Initialization
Po = zeros(L,N);
IDX = zeros(1,K);
TestMatrix = zeros(N);
TestMatrix(1,:) = 1;
for i = 1:N
    idx = floor(rand*K) + 1;
    TestMatrix(2:N,i) = Yr(1:N-1,idx);
    IDX(i) = idx;
end
actualVolume = abs(det(TestMatrix)); % instead of: volumeactual = abs(det(MatrixTest))/(factorial(p-1));
it = 1;
v1 = -1;
v2 = actualVolume;

%% Algorithm
maxit=3*N;
while it<=maxit && v2>v1
    for k=1:N
        for i=1:K
            actualSample = TestMatrix(2:N,k);
            TestMatrix(2:N,k) = Yr(1:N-1,i);
            volume = abs(det(TestMatrix));  % instead of: volume = abs(det(MatrixTest))/(factorial(p-1));
            if volume > actualVolume
                actualVolume = volume;
                IDX(k) = i;
            else
                TestMatrix(2:N,k) = actualSample;
            end
        end
    end
    it = it+1;
    v1 = v2;
    v2 = actualVolume;
end
for i = 1:N
    Po(:,i) = Y(:,IDX(i));
end
end

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%


function Po = VCA(Y,N)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [P,indices,SNRe]=VCA(Y,N)
%
% Vertex Component Analysis algorithm for endmembers estimation in multi/hyperspectral dataset
%  
%
% Inputs
%   Y --> Multi/hyperspectral dataset as 2D matrix (L x K).
%   N --> Number of endmembers to find.
%
% Outputs
%   P --> Matrix of endmembers (L x N).
%
% References
%   J. M. P. Nascimento and J. M. B. Dias, ?Vertex component analysis: A 
% fast algorithm to unmix hyperspectral data,? IEEE Transactions on 
% Geoscience and Remote Sensing, vol. 43, no. 4, apr 2005.
%
% DUCD February/2021
% IICO-FC-UASLP
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialization.
K = size(Y, 2);
L = size(Y, 1);

yMean = mean(Y, 2);
RZeroMean = Y - repmat(yMean, 1, K);
[Ud, ~, ~] = svds(RZeroMean*RZeroMean.'/K, N);
Rd = Ud.'*(RZeroMean);
P_R = sum(Y(:).^2)/K;
P_Rp = sum(Rd(:).^2)/K + yMean.'*yMean;
SNR = abs(10*log10( (P_Rp - (N/L)*P_R) / (P_R - P_Rp) ));

SNRth = 15 + 10*log(N) + 8;
if (SNR > SNRth) 
    d = N;
    [Ud, ~, ~] = svds((Y*Y.')/K, d);
    Yd = Ud.'*Y;
    u = mean(Yd, 2);
    M =  Yd ./ repmat( sum( Yd .* repmat(u,[1 K]) ) ,[d 1]);
else
    d = N-1;
    r_bar = mean(Y.').';
    Ud = pca(Y, d);
    R_zeroMean = Y - repmat(r_bar, 1, K);
    Yd = Ud.' * R_zeroMean;
     c = zeros(N, 1);
    for j=1:K
        c(j) = norm(Yd(:,j));
    end
    c = repmat(max(c), 1, K);
    M = [Yd; c];
end
e_u = zeros(N, 1);
e_u(N) = 1;
A = zeros(N, N);
% idg - Doesnt match.
A(:, 1) = e_u;
I = eye(N);
k = zeros(K, 1);
for i=1:N
    w = rand(N, 1);
    % idg - Oppurtunity for speed up here.
    tmpNumerator =  (I-A*pinv(A))*w;
    %f = ((I - A*pinv(A))*w) /(norm( tmpNumerator ));
    f = tmpNumerator / norm(tmpNumerator);

    v = f.'*M;
    k = abs(v);
    [~, k] = max(k);
    A(:,i) = M(:,k);
    indices(i) = k;
end
if (SNR > SNRth)
    Po = Ud*Yd(:,indices);
else
    Po = Ud*Yd(:,indices) + repmat(r_bar, 1, N);
end
return;
end

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

function [U] = pca(X, d)
    N = size(X, 2);
    xMean = mean(X, 2);
    XZeroMean = X - repmat(xMean, 1, N);     
    [U,~,~] = svds((XZeroMean*XZeroMean.')/N, d);
return;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

function [A_est, time, index]=SVMAX(X,N)     
%=====================================================================
% Programmers: 
% Tsung-Han Chan, E-mail: thchan@ieee.org  
% A. ArulMurugan, E-mail: aareul@ieee.org
% Date: Sept, 2010
%======================================================================
% A implementation of SVMAX
% [A_est time index]=SVMAX(X,N)
%======================================================================
%  Input
%  X is M-by-L data matrix where M is the spectral bands (or observations) and L is the number of pixels (data length).   
%  N is the number of endmembers (or sources).
%----------------------------------------------------------------------
%  Output
%  A_est is M-by-N: estimated endmember signatures (or mixing matrix) obtained by SVMAX.
%  time is the computation time (in secs). 
%  index is the set of indices of the pure pixels identified by SVMAX
%========================================================================

t0 = clock;
[M,L] = size(X);
d = mean(X,2);
U = X-d*ones(1,L);
OPTS.disp = 0;
[C D] = eigs(U*U',N-1,'LM',OPTS);
Xd_t = C'*U;
%=====SVMAX algorithm=========
A_set=[]; Xd = [Xd_t; ones(1,L)]; index = []; P = eye(N);                         
for i=1:N
    [val ind]=max(sum(abs(P*Xd).^2).^(1/2));    
    A_set = [A_set Xd(:,ind)];                            
    P = eye(N) - A_set*pinv(A_set);                       
    index = [index ind];                                        
end
A_est=C*Xd_t(:,index)+d*ones(1,N);
time = etime(clock,t0);
end

%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%

function [M,Up,my,sing_values] = SISAL(Y,p,varargin)

%% [M,Up,my,sing_values] = sisal(Y,p,varargin)
%
% Simplex identification via split augmented Lagrangian (SISAL)
%
%% --------------- Description ---------------------------------------------
%
%  SISAL Estimates the vertices  M={m_1,...m_p} of the (p-1)-dimensional
%  simplex of minimum volume containing the vectors [y_1,...y_N], under the
%  assumption that y_i belongs to a (p-1)  dimensional affine set. Thus,
%  any vector y_i   belongs  to the convex hull of  the columns of M; i.e.,
%
%                   y_i = M*x_i
%
%  where x_i belongs to the probability (p-1)-simplex.
%
%  As described in the papers [1], [2], matrix M is  obtained by implementing
%  the following steps:
%
%   1-Project y onto a p-dimensional subspace containing the data set y
%
%            yp = Up'*y;      Up is an isometric matrix (Up'*Up=Ip)
%
%   2- solve the   optimization problem
%
%       Q^* = arg min_Q  -\log abs(det(Q)) + tau*|| Q*yp ||_h
%
%                 subject to:  ones(1,p)*Q=mq,
%
%      where mq = ones(1,N)*yp'inv(yp*yp) and ||x||_h is the "hinge"
%              induced norm (see [1])
%   3- Compute
%
%      M = Up*inv(Q^*);
%
%% -------------------- Line of Attack  -----------------------------------
%
% SISAL replaces the usual fractional abundance positivity constraints, 
% forcing the spectral vectors to belong to the convex hull of the 
% endmember signatures,  by soft  constraints. This new criterion brings
% robustnes to noise and outliers
%
% The obtained optimization problem is solved by a sequence of
% augmented Lagrangian optimizations involving quadractic and one-sided soft
% thresholding steps. The resulting algorithm is very fast and able so
% solve problems far beyond the reach of the current state-of-the art
% algorithms. As examples, in a standard PC, SISAL, approximatelly, the
% times:
%
%  p = 10, N = 1000 ==> time = 2 seconds
%
%  p = 20, N = 50000 ==> time = 3 minutes
%
%%  ===== Required inputs =============
%
% y - matrix with  L(channels) x N(pixels).
%     each pixel is a linear mixture of p endmembers
%     signatures y = M*x + noise,
%
%     SISAL assumes that y belongs to an affine space. It may happen,
%     however, that the data supplied by the user is not in an affine
%     set. For this reason, the first step this code implements
%     is the estimation of the affine set the best represent
%     (in the l2 sense) the data.
%
%  p - number of independent columns of M. Therefore, M spans a
%  (p-1)-dimensional affine set.
%
%
%%  ====================== Optional inputs =============================
%
%  'MM_ITERS' = double; Default 80;
%
%               Maximum number of constrained quadratic programs
%
%
%  'TAU' = double; Default; 1
%
%               Regularization parameter in the problem
%
%               Q^* = arg min_Q  -\log abs(det(Q)) + tau*|| Q*yp ||_h
%
%                 subject to:ones(1,p)*Q=mq,
%
%              where mq = ones(1,N)*yp'inv(yp*yp) and ||x||_h is the "hinge"
%              induced norm (see [1]).
%
%  'MU' = double; Default; 1
%
%              Augmented Lagrange regularization parameter
%
%  'spherize'  = {'yes', 'no'}; Default 'yes'
%
%              Applies a spherization step to data such that the spherized
%              data spans over the same range along any axis.
%
%  'TOLF'  = double; Default; 1e-2
%
%              Tolerance for the termination test (relative variation of f(Q))
%
%
%  'M0'  =  <[Lxp] double>; Given by the VCA algorithm
%
%            Initial M.
%
%
%  'verbose'   = {0,1,2,3}; Default 1
%
%                 0 - work silently
%                 1 - display simplex volume
%                 2 - display figures
%                 3 - display SISAL information 
%                 4 - display SISAL information and figures
%
%
%
%
%%  =========================== Outputs ==================================
%
% M  =  [Lxp] estimated mixing matrix
%
% Up =  [Lxp] isometric matrix spanning  the same subspace as M
%
% my =   mean value of y
%
% sing_values  = (p-1) eigenvalues of Cy = (y-my)*(y-my)/N. The dynamic range
%                 of these eigenvalues gives an idea of the  difficulty of the
%                 underlying problem
%
%
% NOTE: the identified affine set is given by
%
%              {z\in R^p : z=Up(:,1:p-1)*a+my, a\in R^(p-1)}
%
%% -------------------------------------------------------------------------
%
% Copyright (May, 2009):        JosÈ Bioucas-Dias (bioucas@lx.it.pt)
%
% SISAL is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
%
% More details in:
%
% [1] JosÈ M. Bioucas-Dias
%     "A variable splitting augmented lagrangian approach to linear spectral unmixing"
%      First IEEE GRSS Workshop on Hyperspectral Image and Signal
%      Processing - WHISPERS, 2009 (submitted). http://arxiv.org/abs/0904.4635v1
%
%
%
% -------------------------------------------------------------------------
%
%%
%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 2
    error('Wrong number of required parameters');
end
% data set size
[L,N] = size(Y);
if (L<p)
    error('Insufficient number of columns in y');
end
%%
%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
% maximum number of quadratic QPs
MMiters = 80;
spherize = 'yes';
% display only volume evolution
verbose = 0;
% soft constraint regularization parameter
tau = 1;
% Augmented Lagrangian regularization parameter
mu = p*1000/N;
% no initial simplex
M = 0;
% tolerance for the termination test
tol_f = 1e-2;

%%
%--------------------------------------------------------------
% Local variables
%--------------------------------------------------------------
% maximum violation of inequalities
slack = 1e-3;
% flag energy decreasing
energy_decreasing = 0;
% used in the termination test
f_val_back = inf;
%
% spherization regularization parameter
lam_sphe = 1e-8;
% quadractic regularization parameter for the Hesssian
% Hreg = = mu*I
lam_quad = 1e-6;
% minimum number of AL iterations per quadratic problem 
AL_iters = 4;
% flag 
flaged = 0;

%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MM_ITERS'
                MMiters = varargin{i+1};
            case 'SPHERIZE'
                spherize = varargin{i+1};
            case 'MU'
                mu = varargin{i+1};
            case  'TAU'
                tau = varargin{i+1};
            case 'TOLF'
                tol_f = varargin{i+1};
            case 'M0'
                M = varargin{i+1};
            case 'VERBOSE'
                verbose = varargin{i+1};
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end

%%
%--------------------------------------------------------------
% set display mode
%--------------------------------------------------------------
if (verbose == 3) | (verbose == 4)
    warning('off','all');
else
    warning('on','all');
end

%%
%--------------------------------------------------------------
% identify the affine space that best represent the data set y
%--------------------------------------------------------------
my = mean(Y,2);
Y = Y-repmat(my,1,N);
[Up,D] = svds(Y*Y'/N,p-1);
% represent y in the subspace R^(p-1)
Y = Up*Up'*Y;
% lift y
Y = Y + repmat(my,1,N);   %
% compute the orthogonal component of my
my_ortho = my-Up*Up'*my;
% define another orthonormal direction
Up = [Up my_ortho/sqrt(sum(my_ortho.^2))];
sing_values = diag(D);

% get coordinates in R^p
Y = Up'*Y;


%%
%------------------------------------------
% spherize if requested
%------------------------------------------
if strcmp(spherize,'yes')
    Y = Up*Y;
    Y = Y-repmat(my,1,N);
    C = diag(1./sqrt((diag(D+lam_sphe*eye(p-1)))));
    IC = inv(C);
    Y=C*Up(:,1:p-1)'*Y;
    %  lift
    Y(p,:) = 1;
    % normalize to unit norm
    Y = Y/sqrt(p);
end

%%
% ---------------------------------------------
%            Initialization
%---------------------------------------------
if M == 0
    % Initialize with VCA
    Mvca = VCAsisal(Y,'Endmembers',p,'verbose','off');
    M = Mvca;
    % expand Q
    Ym = mean(M,2);
    Ym = repmat(Ym,1,p);
    dQ = M - Ym; 
    % fraction: multiply by p is to make sure Q0 starts with a feasible
    % initial value.
    M = M + p*dQ;
else
    % Ensure that M is in the affine set defined by the data
    M = M-repmat(my,1,p);
    M = Up(:,1:p-1)*Up(:,1:p-1)'*M;
    M = M +  repmat(my,1,p);
    M = Up'*M;   % represent in the data subspace
    % is sherization is set
    if strcmp(spherize,'yes')
        M = Up*M-repmat(my,1,p);
        M=C*Up(:,1:p-1)'*M;
        %  lift
        M(p,:) = 1;
        % normalize to unit norm
        M = M/sqrt(p);
    end
    
end
Q0 = inv(M);
Q=Q0;


% plot  initial matrix M
if verbose == 2 | verbose == 4
    set(0,'Units','pixels')

    %get figure 1 handler
    H_1=figure;
    pos1 = get(H_1,'Position');
    pos1(1)=50;
    pos1(2)=100+400;
    set(H_1,'Position', pos1)

    hold on
    M = inv(Q);
    p_H(1) = plot(Y(1,:),Y(2,:),'.');
    p_H(2) = plot(M(1,:), M(2,:),'ok');

    leg_cell = cell(1);
    leg_cell{1} = 'data points';
    leg_cell{end+1} = 'M(0)';
    title('SISAL: Endmember Evolution')

end


%%
% ---------------------------------------------
%            Build constant matrices
%---------------------------------------------

AAT = kron(Y*Y',eye(p));    % size p^2xp^2
B = kron(eye(p),ones(1,p)); % size pxp^2
qm = sum(inv(Y*Y')*Y,2);


H = lam_quad*eye(p^2);
F = H+mu*AAT;          % equation (11) of [1]
IF = inv(F);

% auxiliar constant matrices
G = IF*B'*inv(B*IF*B');
qm_aux = G*qm;
G = IF-G*B*IF;


%%
% ---------------------------------------------------------------
%          Main body- sequence of quadratic-hinge subproblems
%----------------------------------------------------------------

% initializations
Z = Q*Y;
Bk = 0*Z;


for k = 1:MMiters
    
    IQ = inv(Q);
    g = -IQ';
    g = g(:);

    baux = H*Q(:)-g;

    q0 = Q(:);
    Q0 = Q;
    
    % display the simplex volume
    if verbose == 1
        if strcmp(spherize,'yes')
            % unscale
            M = IQ*sqrt(p);
            %remove offset
            M = M(1:p-1,:);
            % unspherize
            M = Up(:,1:p-1)*IC*M;
            % sum ym
            M = M + repmat(my,1,p);
            M = Up'*M;
        else
            M = IQ;
        end
        fprintf('\n iter = %d, simplex volume = %4f  \n', k, 1/abs(det(M)))
    end

    
    %Bk = 0*Z;
    if k==MMiters
        AL_iters = 100;
        %Z=Q*Y;
        %Bk = 0*Z;
    end
    
    % initial function values (true and quadratic)
    % f0_val = -log(abs(det(Q0)))+ tau*sum(sum(hinge(Q0*Y)));
    % f0_quad = f0_val; % (q-q0)'*g+1/2*(q-q0)'*H*(q-q0);
    
    while 1 > 0
        q = Q(:);
        % initial function values (true and quadratic)
        f0_val = -log(abs(det(Q)))+ tau*sum(sum(hinge(Q*Y)));
        f0_quad = (q-q0)'*g+1/2*(q-q0)'*H*(q-q0) + tau*sum(sum(hinge(Q*Y)));
        for i=2:AL_iters
            %-------------------------------------------
            % solve quadratic problem with constraints
            %-------------------------------------------
            dq_aux= Z+Bk;             % matrix form
            dtz_b = dq_aux*Y';
            dtz_b = dtz_b(:);
            b = baux+mu*dtz_b;        % (11) of [1]
            q = G*b+qm_aux;           % (10) of [1]
            Q = reshape(q,p,p);
            
            %-------------------------------------------
            % solve hinge
            %-------------------------------------------
            Z = soft_neg(Q*Y -Bk,tau/mu);
            
                 %norm(B*q-qm)
           
            %-------------------------------------------
            % update Bk
            %-------------------------------------------
            Bk = Bk - (Q*Y-Z);
            if verbose == 3 ||  verbose == 4
                fprintf('\n ||Q*Y-Z|| = %4f \n',norm(Q*Y-Z,'fro'))
            end
            if verbose == 2 || verbose == 4
                M = inv(Q);
                plot(M(1,:), M(2,:),'.r');
                if ~flaged
                     p_H(3) = plot(M(1,:), M(2,:),'.r');
                     leg_cell{end+1} = 'M(k)';
                     flaged = 1;
                end
            end
        end
        f_quad = (q-q0)'*g+1/2*(q-q0)'*H*(q-q0) + tau*sum(sum(hinge(Q*Y)));
        if verbose == 3 ||  verbose == 4
            fprintf('\n MMiter = %d, AL_iter, = % d,  f0 = %2.4f, f_quad = %2.4f,  \n',...
                k,i, f0_quad,f_quad)
        end
        f_val = -log(abs(det(Q)))+ tau*sum(sum(hinge(Q*Y)));
        if f0_quad >= f_quad    %quadratic energy decreased
            while  f0_val < f_val;
                if verbose == 3 ||  verbose == 4
                    fprintf('\n line search, MMiter = %d, AL_iter, = % d,  f0 = %2.4f, f_val = %2.4f,  \n',...
                        k,i, f0_val,f_val)
                end
                % do line search
                Q = (Q+Q0)/2;
                f_val = -log(abs(det(Q)))+ tau*sum(sum(hinge(Q*Y)));
            end
            break
        end
    end



end

if verbose == 2 || verbose == 4
    p_H(4) = plot(M(1,:), M(2,:),'*g');
    leg_cell{end+1} = 'M(final)';
    legend(p_H', leg_cell);
end


if strcmp(spherize,'yes')
    M = inv(Q);
    % refer to the initial affine set
    % unscale
    M = M*sqrt(p);
    %remove offset
    M = M(1:p-1,:);
    % unspherize
    M = Up(:,1:p-1)*IC*M;
    % sum ym
    M = M + repmat(my,1,p);
else
    M = Up*inv(Q);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Ae, indice, Rp] = VCAsisal(R,varargin)

% Vertex Component Analysis
%
% [Ae, indice, Rp ]= vca(R,'Endmembers',p,'SNR',r,'verbose',v)
%
% ------- Input variables -------------
%  R - matrix with dimensions L(channels) x N(pixels)
%      each pixel is a linear mixture of p endmembers
%      signatures R = M x s, where s = gamma x alfa
%      gamma is a illumination perturbation factor and
%      alfa are the abundance fractions of each endmember.
%      for a given R, we need to decide the M and s
% 'Endmembers'
%          p - positive integer number of endmembers in the scene
%
% ------- Output variables -----------
% A - estimated mixing matrix (endmembers signatures)
% indice - pixels that were chosen to be the most pure
% Rp - Data matrix R projected.   
%
% ------- Optional parameters---------
% 'SNR'
%          r - (double) signal to noise ratio (dB)
% 'verbose'
%          v - [{'on'} | 'off']
% ------------------------------------
%
% Authors: Jos?Nascimento (zen@isel.pt) 
%          Jos?Bioucas Dias (bioucas@lx.it.pt) 
% Copyright (c)
% version: 2.1 (7-May-2004)
%
% For any comment contact the authors
%
% more details on:
% Jos?M. P. Nascimento and Jos?M. B. Dias 
% "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
% submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
% 
% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

         verbose = 'on'; % default
         snr_input = 0;  % default this flag is zero,
                         % which means we estimate the SNR
         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Looking for input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

         dim_in_par = length(varargin);
         if (nargin - dim_in_par)~=1
            error('Wrong parameters');
         elseif rem(dim_in_par,2) == 1
            error('Optional parameters should always go by pairs');
         else
            for i = 1 : 2 : (dim_in_par-1)
                switch lower(varargin{i})
                  case 'verbose'
                       verbose = varargin{i+1};
                  case 'endmembers'     
                       p = varargin{i+1};
                  case 'snr'     
                       SNR = varargin{i+1};
                       snr_input = 1;       % flag meaning that user gives SNR 
                  otherwise
                       fprintf(1,'Unrecognized parameter:%s\n', varargin{i});
                end %switch
            end %for
         end %if

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initializations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
         if isempty(R)
            error('there is no data');
         else
            [L N]=size(R);  % L number of bands (channels)
                            % N number of pixels (LxC) 
         end                   
               
         if (p<0 | p>L | rem(p,1)~=0),  
            error('ENDMEMBER parameter must be integer between 1 and L');
         end
        
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SNR Estimates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

         if snr_input==0,
            r_m = mean(R,2);      
            R_m = repmat(r_m,[1 N]); % mean of each band
            R_o = R - R_m;           % data with zero-mean 
            [Ud,Sd,Vd] = svds(R_o*R_o'/N,p);  % computes the p-projection matrix 
            x_p =  Ud' * R_o;                 % project the zero-mean data onto p-subspace
            
            SNR = estimate_snr(R,r_m,x_p);
            
            if strcmp (verbose, 'on'), fprintf(1,'SNR estimated = %g[dB]\n',SNR); end
         else   
            if strcmp (verbose, 'on'), fprintf(1,'input    SNR = %g[dB]\t',SNR); end
         end

         SNR_th = 15 + 10*log10(p);
         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choosing Projective Projection or 
%          projection to p-1 subspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

         if SNR < SNR_th,   
                if strcmp (verbose, 'on'), fprintf(1,'... Select the projective proj.\n',SNR); end
                
                d = p-1;
                if snr_input==0, % it means that the projection is already computed
                     Ud= Ud(:,1:d);    
                else
                     r_m = mean(R,2);      
                     R_m = repmat(r_m,[1 N]); % mean of each band
                     R_o = R - R_m;           % data with zero-mean 
         
                     [Ud,Sd,Vd] = svds(R_o*R_o'/N,d);  % computes the p-projection matrix 

                     x_p =  Ud' * R_o;                 % project thezeros mean data onto p-subspace

                end
                
                Rp =  Ud * x_p(1:d,:) + repmat(r_m,[1 N]);      % again in dimension L
                
                x = x_p(1:d,:);             %  x_p =  Ud' * R_o; is on a p-dim subspace
                c = max(sum(x.^2,1))^0.5;
                y = [x ; c*ones(1,N)] ;
         else
                if strcmp (verbose, 'on'), fprintf(1,'... Select proj. to p-1\n',SNR); end
             
                d = p;
                [Ud,Sd,Vd] = svds(R*R'/N,d);         % computes the p-projection matrix 
                
                x_p = Ud'*R;
                Rp =  Ud * x_p(1:d,:);      % again in dimension L (note that x_p has no null mean)
                
                x =  Ud' * R;
                u = mean(x,2);        %equivalent to  u = Ud' * r_m
                y =  x./ repmat( sum( x .* repmat(u,[1 N]) ) ,[d 1]);

          end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VCA algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

indice = zeros(1,p);
A = zeros(p,p);
A(p,1) = 1;

for i=1:p
      w = rand(p,1);   
      f = w - A*pinv(A)*w;
      f = f / sqrt(sum(f.^2));
      
      v = f'*y;
      [v_max, indice(i)] = max(abs(v));
      A(:,i) = y(:,indice(i));        % same as x(:,indice(i))
end
Ae = Rp(:,indice);

return;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of the vca function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Internal functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function snr_est = estimate_snr(R,r_m,x)

         [L, N]=size(R);           % L number of bands (channels)
                                  % N number of pixels (Lines x Columns) 
         [p, N]=size(x);           % p number of endmembers (reduced dimension)

         P_y = sum(R(:).^2)/N;
         P_x = sum(x(:).^2)/N + r_m'*r_m;
         snr_est = 10*log10( (P_x - p/L*P_y)/(P_y- P_x) );
return;    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function z = hinge(y)
%  z = hinge(y)
%
%   hinge function
z = max(-y,0);
end

function z = soft_neg(y,tau)
%  z = soft_neg(y,tau);
%
%  negative soft (proximal operator of the hinge function)

z = max(abs(y+tau/2) - tau/2, 0);
z = z./(z+tau/2) .* (y+tau/2);
end


