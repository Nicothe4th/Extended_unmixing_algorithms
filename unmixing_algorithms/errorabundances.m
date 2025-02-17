function Ea=errorabundances(Ao,A)

N=size(Ao,1);
K=size(Ao,2);
Eaa=zeros(N,1);
for i=1:N
    Error=zeros(N,1);
    for j=1:N
        Aoi=Ao(i,:);
        Aj=A(j,:);
        Error(j)=norm(Aoi-Aj,'Fro')/norm(Aoi,'Fro');
    end
    Eaa(i)=min(Error);
end
Ea=sum(Eaa)/(N);