function model = OvR_SVM(X,t,param,kern,C,kernparam)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%  model = OvR_SVM(X,t,param,kern,C,sig)
% INPUTS:
%   X: input samples matrix (R^{N,P})
%   t: targets array
%   kern: struct kernel with fields:
%       .kernfunction: name kernel,  'lin': linearkernel
%                      and 'rbf': gaussian kernel. Default 'lin'
%       .C: regularization parameters
%   .C,.kernparam (optional): regularization parameters and kernel
%                             function's parameter, respectibity.
% OUTPUTS:
%   model: struct with fields:
%       .models: cell matrox with structs for each binary model training
%       .param: struct with regularization parameter. 
%       .kern: struct kernel
%       .labels: array with name targets
%       .K: number of classes


if nargin == 6 && ~isempty(kernparam)
    kern.param = kernparam;
end

if nargin >= 5 && ~isempty(C)
    param.C = C;
end

if nargin < 4
    kern.kernfunction = 'lin';
    if nargin < 3
        param.C = 1;
        if nargin < 2
            error('Argumentos insuficientes.')
        end
    end
end

labels = unique(t);
K = length(labels); % Number of classes
N = size(X,1);

structs = cell(K,1);

for k = 1:K
    tt = zeros(N,1);
    %% Creating the samples set One vs Rest
    index1 = string(t)==string(labels(k));
    index2 = ~index1;
    tt(index1) = 1;
    tt(index2) = -1;
    if strcmp(lower(kern.kernfunction),{'rbf','gaussian'})
        structs{k} = fitcsvm(X,tt,'KernelFunction','RBF','KernelScale',...
            sqrt(2)*kern.param,'BoxConstraint',param.C);
    else
        structs{k} = fitcsvm(X,tt,'KernelFunction','linear','BoxConstraint',param.C);
    end
    
end

model.models = structs;
model.param = param;
model.kern = kern;
model.labels = labels;
model.K = K;