function model = OvO_ETWSVM(X,t,param,kern,kernparam,c11,c12,c21,c22)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%  model = OvO_ETWSVM(X,t,param,kern,c11,c12,c21,c22) 
% INPUTS:
%   X: input samples matrix (R^{N,P})
%   t: targets array
%   param: struct with scalar regularization parameters as fields:
%       .c11,.c12,.c21,.c22: scalar regularization parameters.
%   kern: struct kernel with fields:
%       .kernfunction: name kernel,  'lin': linearkernel
%                      and 'rbf': gaussian kernel. Default 'lin'
%       .E: Transformated matrix of mahalanobis distance
%           R^{PxQ} Q<=P. Default indenty matrix with Q=P.
%       .param: scalar array with kernel's parameters
%   kernparam: numeric array with kernel parameters
%   c11, c12, c21, c22 (optional):  scalar regularization parameter for
%                                   parfor. Default all fixed in one value.
% OUTPUTS:
%   model: struct with fields:
%       .models: cell matrox with structs for each binary model training
%       .param: struct with regularization parameters. 
%       .kern: struct kernel
%       .labels: array with name targets
%       .K: number of classes
%       .nModels: number of the total models K(K-1)/2
%       .targets: cell array with the differents combitations between
%                 classes

%%
if nargin>=6 && ~isempty(c11)
    param.c11 = c11;
end
if nargin>=7 && ~isempty(c12)
    param.c12 = c12;
end
if nargin>=8 && ~isempty(c21)
    param.c21 = c21;
end
if nargin==9 && ~isempty(c22)
    param.c22 = c22;
end

if nargin>= 5 && ~isempty(kernparam)
    kern.param = kernparam;
end

if ~any(strcmp(lower(fieldnames(param)),'typeclassifier'))
    param.typeclassifier = 'multiclass';
end

if nargin < 4
    kern.kernfunction = 'lin';
    if nargin < 3
        param.c11 = 1;
        param.c12 = 1;
        param.c21 = 1;
        param.c22 = 1;
        if nargin < 2
            error('Argumentos insuficientes.')
        end
    end
end

labels = unique(t);
K = length(labels); % Number of classes
% N = size(X,1);
nModels = 0.5*K*(K-1); 
structs = cell(nModels,2);
targets = cell(nModels,1);
ij = nchoosek((1:K)',2); % All possible combinations between classes

for k = 1:nModels
    %% Creating the samples set One vs Rest
    %% hyperplane ij
    index1 = string(t)==string(labels(ij(k,1)));
    index2 = string(t)==string(labels(ij(k,2)));
    targets{k} = ij(k,:);
    X1 = X(index1,:);
    X2 = X(index2,:);
    tt = [ones(size(X1,1),1);-1*ones(size(X2,1),1)];
    XX = [X1;X2];
    structs{k,1} = ETWSVMTraining(XX,tt,param,kern);
    %% hyperplane ji
    X1 = X(index2,:);
    X2 = X(index1,:);
    tt = [ones(size(X1,1),1);-1*ones(size(X2,1),1)];
    XX = [X1;X2];
    structs{k,2} = ETWSVMTraining(XX,tt,param,kern);
end

model.models = structs;
model.param = param;
model.kern = kern;
model.labels = labels;
model.targets = targets;
model.K = K;
model.nModels = nModels;