function [Bb,Bw] = KNNSampling(A,B,K)
%% find the k neigbors of A in B, and the k neigbors of A with-in class
% A is pos, B is neg, K is a parameter struct.
% Author: Weijie Chen
%         wjcper2008@126.com


kb = K.kb;kb1=K.kb1;
kw = K.kw;kw1 = K.kw1;
%kb1 = 5;kw1 = 10;
options = struct('NN', kb, ...
             'GraphDistanceFunction', 'euclidean', ... 
             'GraphWeights', 'heat', ...
             'GraphWeightParam', 0);
n=size(A,1);
m=size(B,1);
p=1:(options.NN); 

%% KNN construct Bb
if n<500 % block size: 500
    step=n;
else
	step=500;
end

idy=zeros(n*options.NN,1);
DI=zeros(n*options.NN,1);
t=0;
s=1;

for i1=1:step:n    
    t=t+1;
    i2=i1+step-1;
    if (i2>n) 
        i2=n;
    end

    Xblock=A(i1:i2,:);
    dt=feval(options.GraphDistanceFunction,Xblock,B);
    [Z,I]=sort(dt,2);
	 	    
    Z=Z(:,p)'; % it picks the neighbors from 1nd to NNth
    I=I(:,p)'; % it picks the indices of neighbors from 2nd to NN+1th
    [g1,g2]=size(I);
    idy(s:s+g1*g2-1)=I(:);
    DI(s:s+g1*g2-1)=Z(:);
    s=s+g1*g2;
end 

I=repmat((1:n),[options.NN 1]);
I=I(:);

switch options.GraphWeights    
    case 'distance' 
        W=sparse(I,idy,DI,n,n);
    
    case 'binary'
        W=sparse(I,idy,1,n,m);

    case 'heat'
        if options.GraphWeightParam==0 % default (t=mean edge length)
            t=mean(DI(DI~=0)); % actually this computation should be
                               % made after symmetrizing the adjacecy
                               % matrix, but since it is a heuristic, this
                               % approach makes the code faster.
        else
            t=options.GraphWeightParam;
        end
        W=sparse(I,idy,exp(-DI.^2/(2*t*t)),n,m);

    otherwise
        error('Unknown weight type');
end

%% Bb
Bb = B(sum(W'~=0,2) >kb1,:);


%% KNN construct Bw
options.NN = kw;
p=2:(options.NN+1);

if m<500 % block size: 500
    step=m;
else
	step=500;
end

idy=zeros(m*options.NN,1);
DI=zeros(m*options.NN,1);
t=0;
s=1;

for i1=1:step:m    
    t=t+1;
    i2=i1+step-1;
    if (i2>m) 
        i2=m;
    end

    Xblock=B(i1:i2,:);
    dt=feval(options.GraphDistanceFunction,Xblock,B);
    [Z,I]=sort(dt,2);
	 	    
    Z=Z(:,p)'; % it picks the neighbors from 1nd to NNth
    I=I(:,p)'; % it picks the indices of neighbors from 2nd to NN+1th
    [g1,g2]=size(I);
    idy(s:s+g1*g2-1)=I(:);
    DI(s:s+g1*g2-1)=Z(:);
    s=s+g1*g2;
end 

I=repmat((1:m),[options.NN 1]);
I=I(:);

switch options.GraphWeights    
    case 'distance' 
        W=sparse(I,idy,DI,m,m);
    
    case 'binary'
        W=sparse(I,idy,1,m,m);

    case 'heat'
        if options.GraphWeightParam==0 % default (t=mean edge length)
            t=mean(DI(DI~=0)); % actually this computation should be
                               % made after symmetrizing the adjacecy
                               % matrix, but since it is a heuristic, this
                               % approach makes the code faster.
        else
            t=options.GraphWeightParam;
        end
        W=sparse(I,idy,exp(-DI.^2/(2*t*t)),m,m);

    otherwise
        error('Unknown weight type');
end

W = min(W,W');
Bw = B(sum(W~=0,2) >kw1,:);
end

function D = euclidean(A,B)
    if (size(A,2) ~= size(B,2))
        error('A and B must be of same dimensionality.');
    end

    if (size(A,2) == 1) % if dim = 1...
        A = [A, zeros(size(A,1),1)];
        B = [B, zeros(size(B,1),1)];
    end
    %boosting product
    a = dot(A,A,2);
    b = dot(B,B,2);
    ab=A*B';
    D = real(sqrt(bsxfun(@plus,a,b')-2*ab));
end