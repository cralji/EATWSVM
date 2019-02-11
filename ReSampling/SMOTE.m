function Synthetic=SMOTE(Samples,N,k)
%%% Samples : samples matrix Txp
%%% N: % de la cantidad de muestras sinteticas
T=size(Samples,1);

if N<100
    Taux=floor((N/100)*T);
    N=100;
    index=randperm(T,Taux);
    Samples=Samples(index,:);
    T=Taux; 
    clear Taux index;
end

N=int16(N/100);
newindex=0;
Synthetic=zeros(T*N,size(Samples,2));

%% KNN 
dist=pdist2(Samples,Samples);
[~,ind]=sort(dist,2);
if k>=T
    ind_knn=ind(:,2:end); 
else
    ind_knn=ind(:,2:k+1); 
end

clear ind dist
%% Generate synthetic samples 
for i=1:T
    nnarray=ind_knn(i,:);
    Synthetic((i-1)*N+1:N*i,:)=Populate(Samples,N,i,nnarray);
end

%% fuction Generate synthetic samples 
function Synthetic=Populate(Samples,N,i,nnarray)
%%% Samples : samples matrix Txp
k=length(nnarray);
Synthetic=zeros(N,size(Samples,2));
j=1;
while N~=0
    
    nn=randperm(k,1); % random nearest neighbors
    
    for attr=1:size(Samples,2)
        dif=Samples(nnarray(nn),attr)-Samples(i,attr);
        gap=rand; %random 
        Synthetic(j,attr)=Samples(i,attr)+gap*dif;
    end
    N=N-1; j=j+1 ;
end