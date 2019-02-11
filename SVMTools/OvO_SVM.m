function model = OvO_SVM(X,t,param,kern,C,kernparam)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%  model = OvO_SVM(X,t,param,kern,C,kernparam) 
% INPUTS:
%   X: input samples matrix (R^{N,P})
%   t: targets array
%   param: estruct with regularization parameters
%        .C: regularization parameter
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
%       .nModels: number of the total models K(K-1)/2
%       .targets: cell array with the differents combitations between
%                 classes
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
nModels = 0.5*K*(K-1); 
structs = cell(nModels,1);
targets = cell(nModels,1);
ij = nchoosek((1:K)',2); % All possible combinations between classes

for k = 1:nModels
    %% Creating the samples set One vs Rest
    index1 = string(t)==string(labels(ij(k,1)));
    index2 = string(t)==string(labels(ij(k,2)));
    targets{k} = ij(k,:);
    X1 = X(index1,:);
    X2 = X(index2,:);
    tt = [ones(size(X1,1),1);-1*ones(size(X2,1),1)];
    XX = [X1;X2];
    if strcmp(lower(kern.kernfunction),{'rbf','gaussian'})
        structs{k} = fitcsvm(XX,tt,'KernelFunction','RBF','KernelScale',...
            sqrt(2)*kern.param,'BoxConstraint',param.C);
    else
        structs{k} = fitcsvm(XX,tt,'KernelFunction','linear','BoxConstraint',param.C);
    end
end
model.models = structs;
model.param = param;
model.kern = kern;
model.labels = labels;
model.targets = targets;
model.K = K;
model.nModels = nModels;