function [P,A,S,Yh,V,Ji]=ESSEAE(Yo,N,parameters,Po)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [P,A,S,Yh,V,J]=ESSEAE(Y,N,parameters,Po)
%
% Estimation by Extended Semi-Supervised End-member and Abundance Extraction (ESSEAE)
% Algorithm and Linear Mixture Model 
%
% Based on --> D. U. Campos-Delgado et al., "Extended Blind End-Member and Abundance 
%   Extraction for Biomedical Imaging Applications," IEEE Access, 
%   vol. 7, pp. 178539-178552, 2019, doi: 10.1109/ACCESS.2019.2958985.
%
%
% Input Arguments
%
%   Y = matrix of measurements (LxK)
%       (L --> # of Spectral Channels or Time Samples, 
%        K --> # of Spatial Mesasurements)
%   N = order of linear mixture model
%   parameters = 9x1 vector of hyper-parameters in EBEAE methodology
%              = [initicond rho lambda lm epsilon maxiter downsampling  ...
%                      parallel display]
%       initcond = initialization of end-members matrix {1,2,3}
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
%       lm --> weight parameter in estimation of sparse noise component >=0
%            (default lm=0.01)
%       epsilon = threshold for convergence in ALS method 
%                 (default epsilon=1e-3); 
%       maxiter = maximum number of iterations in ALS method
%                 (default maxiter=20);
%       downsampling = percentage of reduced random downsampling in end-member 
%                      estimation [0,1) (default downsampling=0.5);
%       parallel = implement parallel computation of abundances (0 -> NO or 1 -> YES)
%                  (default parallel=0);
%       display = show progress of iterative optimization process (0 -> NO or 1 -> YES)
%                 (default display=0);
%   Po = fixed end-members for estimation (LxNo) (No <=N)
%
% Output Arguments
%
%   P  --> complete matrix of end-members (LxN)
%   A  --> abundances matrix (NxK)
%   S --> scaling vector (Kx1)
%   Yh--> estimated matrix of measurements (LxK) --> Yh = P*A*diag(S)
%   J --> vector with cost function values during iterative process
%
% Nicolas Mendoza Chavarria & Daniel Ulises Campos Delgado
% FC-UASLP & ULPGC
% Version: October/2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default hyper-parameters of EBEAE algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NUMERROR

initcond=1;
rho=0.1;
lambda=0;
epsilon=1e-3;
maxiter=20;
lm=0.01;
downsampling=0.5;
parallel=0;
display=0;
Nf=0;

NUMERROR=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check concistency of input arguments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<4
    disp('The number of inout arguments is lower than expected!!!');
    return;
end
if nargin==0
    disp('The measurement matrix Y has to be used as argument!!');
    return;
elseif nargin==1
    N=2;
end
if nargin==3 || nargin==4 
    if length(parameters)~= 9
        disp('The length of parameters vector is not 8 !!');
        disp('Default values of hyper-parameters are used instead');
    else
        initcond=round(parameters(1));
        rho=parameters(2);
        lambda=parameters(3);
        lm=parameters(4);
        epsilon=parameters(5);
        maxiter=parameters(6);
        downsampling=parameters(7);
        parallel=parameters(8);
        display=parameters(9);
       if initcond~=1 && initcond~=2 && initcond~=3 && initcond~=4 && initcond~=5 && initcond~=6 && initcond~=7 && initcond~=8
            disp('The initialization procedure of end-members matrix is 1 to 8!');
            disp('The default value is considered!');
            initcond=1;
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
        if lm<0
            disp('The sparse noise weight has to be positive');
            disp('The default value is considered!');
            lm=0.01;
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
if nargin==4 
    
    if ~ismatrix(Po)
        disp('The initial end-members Po must be a matrix !!');
        disp('The initialization is considered by VCA of the input dataset');
        initcond=6;
        Nf=0;
    else
        if size(Po,1)==size(Yo,1) && size(Po,2)==N
            initcond=0;
            Nf=N;
        elseif size(Po,1)~=size(Yo,1)
            disp('The number of spectral channels in Po is not correct !')
            disp('The initialization is considered by VCA of the input dataset');
            initcond=6;
            Nf=0;
        elseif size(Po,1)==size(Yo,1) && size(Po,2)<N
            Nf=size(Po,2);
            Pf=Po;
        elseif size(Po,2) > N
            disp('The number of columns in Po is larger than the order of the linear model!!');
            disp('The initialization is considered by VCA of the input dataset');
            initcond=6;
            Nf=0;
        end
    end
end
if nargin>4
    disp('The number of input arguments is 4 maximum');
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
    disp('Number of spatial measurements has to be larger than the number of spectral channels or time samples!');
    return;
end

I=1:K;
Kdown=round(K*(1-downsampling));
Is=randperm(K,Kdown);
Y=Yo(:,Is);
Vo=zeros(L,K);
Vm=Vo(:,Is);

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

if initcond>0 && Nf<N
    if display==1
        disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
        disp('Estimating initial conditions of free end-members');
        disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    end
    Po=initPo(Yo,Ym,initcond,N,Nf,Pf);
else
    if display==1
        disp('The end-members matrix Po is assumed fixed!');
    end
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

if display==1
        disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
        disp('Hybrid EBEAE Linear Unmixing');
        disp(['Model Order =' num2str(N)]);
        if initcond==0
            disp('The end-members matrix Po is fixed by the user');
        elseif initcond==1
            disp('Updatable Po is constructed based on the maximum cosine difference from mean measurement'); 
        elseif initcond==2
            disp('Updatable Po is constructed based on the maximum and minimum energy, and largest difference from them');
        elseif initcond==3
            disp('Updatable Po is constructed based on the PCA selection + Rectified Linear Unit');
        elseif initcond==4
            disp('Updatable Po is constructed based on the ICA selection (FOBI) + Rectified Linear Unit');
        elseif initcond==5
            disp('Updatable Po is constructed based on N-FINDR endmembers estimation by Winter (1999)');
        elseif initcond==6
            disp('Updatable Po is constructed based on Vertex Component Analysis by Nascimento and Dias (2005)');
        elseif initcond==7
            disp('Updatable Po is constructed based on Simplex Volume Maximization by Chan et al. (2011)');
        elseif initcond==8
            disp('Updatable Po is constructed based on Simplex identification via split augmented Lagrangian by Bioucas-Dias (2009)');
        end
end

while (Jp-J)/Jp >= epsilon && iter < maxiter && initcond~=0 && NUMERROR==0
    
    Am = abundance(Ym,P,Vm,lambda,parallel);
    Pp=P;
    if NUMERROR==0 
        P = hybridEndmember(Ym,Am,P,Vm,Nf,rho); 
    end
    Vm = sparsenoise(Ym,P,Am,lm);
       
    
    Jp=J;
    recidual = Ym - (P*Am)-Vm;
    J=norm(recidual,'fro'); 
    Ji(iter)=(Jp-J)/Jp;
    if J > Jp
        P=Pp; %break;
    end
    
    
    
    if display ==1
        disp(['Number of iteration =' num2str(iter)]);
        disp(['Percentage Estimation Error =' num2str(100*J/NYm) '%']);
    end
    iter=iter+1;
    
end
if NUMERROR==0
  
    if initcond==0  
        V=zeros(size(Ymo));
        J=1e5;
        Jp=1e6;
        Ji=[];
        iter=1;
        while (Jp-J)/Jp >= epsilon && iter < maxiter && NUMERROR==0
            A=abundance(Ymo,P,V,lambda,parallel);  
            V=sparsenoise(Ymo,P,A,lm);
            Jp=J;
            J=norm(Ymo-P*A-V,'fro');
            Ji(iter)=J;
            iter=iter+1;
        end
    
    else
        Ins=setdiff(I,Is); 
        Ymos=Ymo(:,Ins);
        Vms=zeros(size(Ymos));
        J=1e5;
        Jp=1e6;
        iter=1;
        while (Jp-J)/Jp >= epsilon && iter < maxiter && NUMERROR==0
            Ams = abundance(Ymos,P,Vms,lambda,parallel);
            Vms=sparsenoise(Ymos,P,Ams,lm);
            Jp=J;
            J=norm(Ymos-P*Ams-Vms,'fro');
            iter=iter+1;
        end
        A=[Am Ams];
        V=[Vm Vms];
        II=[Is Ins];
        [~,Index]=sort(II);
        A=A(:,Index);
        V=V(:,Index);
    end

    ElapTime=toc;
    if display ==1
            disp(['Elapsep Time =' num2str(ElapTime)]);
    end
    S=mYmo';
    V=V.*repmat(mYmo,[L,1]);
    AA = A.*repmat(mYmo,[N,1]);
    Yh=P*AA + V;
else
    disp('Please revise the problem formulation, not reliable results');
    sound(0.1*sin(2*pi*(1:1000)/10))
    P=[];
    S=[];
    A=[];
    Yh=[];
    V=[];
end
end

%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%

function Po=initPo(Yo,Ym,initcond,N,Nf,Pf)


[L,K]=size(Yo);
Nu=N-Nf;

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
        K=size(Yo,2);
        II=1:K;
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
    [V,~,~] = svd((repmat(sum(Yo_w.*Yo_w,1),L,1).*Yo_w)*Yo_w');
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

Po=replaceEM(Pf,Po);
end

%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%

function Pnew = replaceEM(Pfix, Po)

Nf = size(Pfix,2);
N = size(Po,2);
dsam=zeros(N,1);
for i = 1:N
    for j=1:Nf
        sam = spectralAngleMapper(Po(:,i), Pfix(:,j));
        dsam(i) = dsam(i) +  sam;
    end
end
[~, sortedIdx] = sort(dsam, 'descend');
Pfar = Po(:,sortedIdx(1:(N-Nf)));
Pnew = [Pfix, Pfar];
end

function d = spectralAngleMapper(v1, v2)
    % Calculate spectral angle mapper distance between two vectors
    cos_dist = dot(v1, v2) / (norm(v1) * norm(v2));
    
    % Clip the cosine distance to be within [-1, 1] to avoid numerical issues
    cos_dist = min(max(cos_dist, -1), 1);
    
    % Compute the spectral angle (in degrees) using the inverse cosine
    d = acosd(cos_dist);
end


%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%

function A = abundance(Y,P,V,lambda,parallel)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% A = abundance(Y,P,V,lambda,parallel)
%
% Estimation of Optimal Abundances in Linear Mixture Model
%
% Input Arguments
% Y = matrix of measurements
% P = matrix of end-members
% V = matrix of sparse noise 
% lambda =  entropy weight in abundance estimation \in (0,1)
% parallel = implementation in parallel of the estimation
%
% Output Argument
% A = abundances matrix 
%
% Daniel U. Campos-Delgado
% FC-UASLP / ULPGC
% Version: May/2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check arguments dimensions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NUMERROR

[L,K]=size(Y);
N=size(P,2);
A=zeros(N,K);

if size(P,1) ~= L
    disp('ERROR: the number of rows in Y and P does not match');
    NUMERROR=1;
    sound(0.1*sin(2*pi*(1:1000)/10))
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute fixed vectors and matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c = ones(N,1);
d = 1;
Go=P'*P;
em=eye(N);
lmin=min(eig(Go));
G=Go-em*lmin*lambda;
while rcond(G)<1e-6
    lambda=lambda/2;
    G=Go-em*lmin*lambda;
    if lambda<1e-6
       disp('Unstable numerical results in abundances estimation, update lambda!!');
       NUMERROR=1;
       sound(0.1*sin(2*pi*(1:1000)/10))
       return;
    end
end
Gi=inv(G);
T1=Gi*c;
T2=c'*T1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start Computation of Abundances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if parallel==1
    
    parfor k=1:K

        vk=V(:,k);
        yk=Y(:,k);
        sk=yk-vk;
        bk=P'*sk;
        byk=yk'*yk;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute Optimal Unconstrained Solution
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        dk=(bk'*T1-(1-sum(vk)))/T2;
        ak = Gi*(bk-dk*c);

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

                Gamma(1:N,1:N) = G/byk;
                Gamma(1:N,N+1) = c;
                Gamma(N+1,1:N) = c';

                cont = 0;
                for i = 1:N
                    if(Iset(i)~= 0)
                        cont = cont + 1;
                        ind = i; 
                        Gamma(ind,N+1+cont) = 1;
                        Gamma(N+1+cont,ind) = 1;   
                    end
                end

                Beta(1:N) = bk/byk;
                Beta(N+1) = d-sum(vk);
                delta = Gamma\Beta;
                ak = delta(1:N);
                ak(abs(ak)<1e-9) = 0;
            end    
        end
        A(:,k) = single(ak); 
    end
    
else
    
    for k=1:K

        vk=V(:,k);
        yk=Y(:,k);
        sk=yk-vk;
        bk=P'*sk;
        byk=yk'*yk;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute Optimal Unconstrained Solution
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        dk=(bk'*T1-(1-sum(vk)))/T2;
        ak = Gi*(bk-dk*c);

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

                Gamma(1:N,1:N) = G/byk;
                Gamma(1:N,N+1) = c;
                Gamma(N+1,1:N) = c';

                cont = 0;
                for i = 1:N
                    if(Iset(i)~= 0)
                        cont = cont + 1;
                        ind = i; 
                        Gamma(ind,N+1+cont) = 1;
                        Gamma(N+1+cont,ind) = 1;   
                    end
                end

                Beta(1:N) = bk/byk;
                Beta(N+1) = d-sum(vk);
                delta = Gamma\Beta;
                ak = delta(1:N);
                ak(abs(ak)<1e-9) = 0;
            end    
        end
        A(:,k) = single(ak); 
    end
    
end
end

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%


function Pn = hybridEndmember(Y,A,P,V,Nf,rho)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  Pn = hybridEndmember(Y,A,P,V,Nf,rho)
%
% Estimation of Optimal End-members in Linear Mixture Model
%
% Input Arguments
%
%   Y --> Matrix of measurements
%   A -->  Matrix of abundances
%   P --> Matrix of endmembers
%   V --> Matrix of sparse noise
%   Nf --> number of fixed endmembers
%   rho --> Weighting factor of regularization term
% 
% Output Arguments
%
%   Pn --> Matrix of end-members
%
% Nicolas Mendoza Chavarria & Daniel Ulises Campos-Delgado
% FC-UASLP / ULPGC
% Version: May/2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check arguments dimensions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NUMERROR

[N,K]=size(A);
L=size(Y,1);
Nu=N-Nf;
R=sum(N - (1:(N-1)));
W=repmat((1./K./sum(Y.^2,1))',1,Nu);

if size(Y,2) ~= K
    disp('ERROR: the number of columns in Y and A does not match');
    NUMERROR=1;
    sound(0.1*sin(2*pi*(1:1000)/10));
    return;
end

Pf=P(:,1:Nf);
Af=A(1:Nf,:);
Au=A(Nf+1:N,:);
X=Y-Pf*Af-V;
O1=-2*ones(Nu,Nf);
O2=N*eye(Nu) - ones(Nu,Nu);   
LU1 = ones(L,Nu);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct Optimal End-members Matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T0=Au*(W.*Au') + rho*O2'/R;
while rcond(T0)<1e-6
   rho=rho/10;
   T0=Au*(W.*Au') + rho*O2'/R;
   if rho<1e-6
       disp('Unstable numerical results in end-members estimation, update rho!!');
       NUMERROR=1;
       return;
   end
end
V = eye(Nu)/T0;
T2 = X*(W.*Au')-((rho/(2*R))*Pf*O1');
T1 = eye(L) - (1/L)*(ones(L));
T3 = (1/L)*LU1;
P_est = T1*T2*V + T3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate and Project Negative Elements 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P_est(P_est<0) = 0;
P_est(isnan(P_est))=0;
P_est(isinf(P_est))=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Normalize Optimal Solution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Pnn=[Pf P_est];
Pn=Pnn./repmat(sum(Pnn,1),L,1);
end

function V = sparsenoise(Y,P,A,lm)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  V = sparsenoise(Y,P,A,lm)
%
% Estimation of Sparse Noise Component
%
% Input Arguments
% Y --> Matrix of measurements
% P --> Matrix of end-members
% A -->  Matrix of abundances
% lm --> weight on sparse noise estimation
% 
% Output Arguments
% 
% V --> Matrix of sparse noise
%
% Daniel Ulises Campos-Delgado
% FC-UASLP & ULPGC
% Version: May/2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

E=Y-P*A;
L=size(Y,1);
Ye=repmat(sum(Y.^2,1),[L,1]);
V=sign(E).*max(0,abs(E)-lm*Ye);
V=max(0,V);
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
% [1] Winter, M. E., �N-FINDR: an algorithm for fast autonomous spectral 
%     end-member determination in hyperspectral data�, presented at the 
%     Imaging Spectrometry V, Denver, CO, USA, 1999, vol. 3753, p�gs. 266-275.
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
    %Ud = Ud(:, 1:d);
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
% Copyright (May, 2009):        Jos� Bioucas-Dias (bioucas@lx.it.pt)
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
% [1] Jos� M. Bioucas-Dias
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

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
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
      [v_max indice(i)] = max(abs(v));
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

         [L N]=size(R);           % L number of bands (channels)
                                  % N number of pixels (Lines x Columns) 
         [p N]=size(x);           % p number of endmembers (reduced dimension)

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

