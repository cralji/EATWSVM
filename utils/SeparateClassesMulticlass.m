function [Xsort,tsort,labels]=SeparateClassesMulticlass(X,t,op)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   [Xsort,tsort,labels]=SeparacionClasesMulticlass(X,t,op)
% INPUTS:
%   X: Samples matrix R^{NxP}
%   t: targets vector R^{Nx1}
%   op: boolean scalar, where if it is fixed in 1 the all classes are
%       under-sampling to minority class. On the other hand, if this is
%       fixed in 0, the classes doesn't undersampling.
% OUTPUTS:
%   Xsort: samples matrix sorted.
%   tsort: target array sorted
%   labels : cell or number array with the targets exists in t.
if nargin < 3
    op = 1;
    if nargin<2
        error('missing arguments')
    end
end
labels = unique(t);
K = length(labels);
N = zeros(K,1);
for k=1:K
    N(k) = numel(find(string(t)==string(labels(k))));
end
[nm,indmin] = min(N);
index = string(t)==string(labels(indmin));
X1 = X(index,:);
% t1 = t(index);
Xsort = [];
tsort = [];


for k = 1:K
    if op==1
        if k~=indmin
            index = find(string(t)==string(labels(k)));
            Xaux = [X1;X(index,:)];
            taux = [ones(nm,1);-1*ones(length(index),1)];
            [Xus,~] = US_vnn(Xaux,taux,8);
            Xsort = [Xsort ; Xus];
            tsort = [tsort ; k*ones(size(Xus,1),1)];
        else
            Xsort = [Xsort;X1];
            tsort = [tsort;indmin*ones(nm,1)];
        end
    elseif op==0
         index2 = find(string(t)==string(labels(k)));
         Xsort = [Xsort;X(index2,:)];
         tsort = [tsort ; k*ones(size(index2,1),1)];
    end
end