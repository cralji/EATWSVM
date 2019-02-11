function [Xus,tus,index]=US_vnn(X,t,k,m,rho)
% Algorithm extracted from Yuan-HaiShao,Wei-Jie Chena, Jing-Jing Zhang, 
% Zhen Wang,Nai-Yang Deng, An efficient weighted Lagrangian twin support 
% vector machine for imbalanced data classification
% @CopyRight Cristian Alfonso Jimenez Castaño e-mail:
% craljimenez@utp.edu.co
% USAGE:
%   [Xus,tus] = US_vnn(X,t,k,m)
% INPUTS:
%     X : samples matrix \in R^{NxP}. N is the number of samples and
%         P is the dimensionality.
%     t : target array. \in R^{N,1}
%     k : nearest neighbors. Default (10)
%     m : desired rate. Default (1)
%     rho: scalar weighted. Default (1)
% OUTPUTS:
%     Xus : undersampling samples
%     tus : undersampling samples targets.
%     index : samples index undersampling
    
if nargin < 5
    rho = 1;
    if nargin < 4
        m = 1;
        if nargin <3 
            k = 10;
            if nargin < 2
                error('missing arguments')
            end
        end
    end
end
%% identification of the majority class
C = unique(t);
Card = [numel(find(t==C(1))) numel(find(t==C(2)))];
[~,majo] = max(Card);
%% separation of the majority class and minority one.
index = find(t==C(majo));
Xmajo = X(t==C(majo),:);
Xmino = X(t~=C(majo),:);
tmino = t(t~=C(majo));
tmajo = t(t==C(majo));
nmajo = size(Xmajo,1);
nmino = size(Xmino,1);
%% number of samples to undersampling the majority samples
M = m*nmino;
%% warnings
if nmino==nmajo || M > nmajo
    warning('Balanded classes or disproportionate rate.')
    Xus = X;
    tus = t;
else

    %% apply algorithm 
    D = pdist2(Xmajo,Xmajo);
    [~,ind] = sort(D,2);    
    ind = ind(:,2:k+1);     
    U = zeros(nmajo,nmajo);
    for i=1:nmajo
        [idrow,~] = find(i==ind);
        inter  =  intersect(idrow,ind(i,:));
        U(i,inter) = rho;
    end
    u = sum(U,2);
    [~,ind] = sort(u,'descend');

    Xaux = Xmajo(ind(1:M),:);
    taux = tmajo(ind(1:M));
    Xus = [Xmino;Xaux];
    tus = [tmino;taux];
    index = index(ind(1:M));
end    