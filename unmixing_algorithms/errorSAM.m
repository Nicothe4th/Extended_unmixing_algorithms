function Ep=errorSAM(Po,P)

N=size(Po,2);
Epp=zeros(N,1);

for i=1:N
    Error=zeros(N,1);
    for j=1:N
        Poi=Po(:,i)/sum(Po(:,i));
        Pj=P(:,j)/sum(P(:,j));
        Error(j)=acos( (Poi'*Pj)/(norm(Poi)*norm(Pj)) );
    end
    Epp(i)=min(real(Error));
end
Ep=sum(Epp)/N;