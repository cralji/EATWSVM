function [t_est,structTrain] = SVMTraining(X,t,XTest,kern,sig,c)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%   [t_est,structTrain] = SVMTraining(X,t,XTest,kern,sig,c)
% INPUTS:
%   X: samples matrix for training R^{NxP} 
%   t: targets array for training R^{Nx1} 
%   Xtest: samples matrix for estimate targets
%   kern: kernel struct with fields:
%       .C: regularization parameters
%       .param: kernel function's parameters
%       .function: kernel function's name. 'linear'(Default) and 'rbf'
% OUTPUTS:
%   t_est: estimates targets array
%   structTrain: struct with estimated model. See >> doc svmtrain
warning('off','all');
if nargin == 6 && ~isempty(c)
    kern.C = c;
end
if nargin >= 5 && ~isempty(sig)
    kern.param = sig;
end
opts=statset('Display','off','MaxIter',10000000000);
nameKern=kern.function;
C = kern.C*ones(size(X,1),1);
param = kern.param;

if strcmp(nameKern,'linear') 
    structTrain=svmtrain(X,t,'method','SMO','kernel_function',nameKern,...
    'boxconstraint',C,'tolkkt',1e-3,'options',opts); 
else
    if strcmp(nameKern,'rbf')
        nameParam='rbf_sigma';
    end
    if strcmp(nameKern,'polynomial')
        nameParam='polyorder';
    end
        structTrain=svmtrain(X,t,'method','SMO','kernel_function',nameKern,...
        nameParam,param,'boxconstraint',C,'tolkkt',1e-3,'options',opts);
end
%     H = exp(-pdist2(XTrain,XTrain).^2./(2*param^2));
t_est = svmclassify(structTrain,XTest,'Showplot',false);