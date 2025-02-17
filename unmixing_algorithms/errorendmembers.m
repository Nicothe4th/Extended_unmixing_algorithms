function Ep=errorendmembers(Po,P)

N=size(Po,2);
Epp=zeros(N,1);
for i=1:N
    Error=zeros(N,1);
    for j=1:N
        Poi=Po(:,i)/sum(Po(:,i));
        Pj=P(:,j)/sum(P(:,j));
        Error(j)=norm(Poi-Pj,'Fro')/ norm(Poi,'Fro');
    end
    Epp(i)=min(Error);
end
Ep=sum(Epp)/N;