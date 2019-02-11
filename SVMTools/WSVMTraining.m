function [t_est,structTrain] = WSVMTraining(X,t,XTest,kern,sig,c)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%   [t_est,structTrain] = WSVMTraining(X,t,XTest,kern,sig,c)
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
warning('off','all')
if nargin == 6 && ~isempty(c)
    kern.C = c;
end
if nargin >= 5 && ~isempty(sig)
    kern.param = sig;
end
opts=statset('Display','off','MaxIter',100000000000000);

nameKern=kern.function;
C = kern.C;
param=kern.param;



if strcmp(nameKern,'linear')
    structTrain=svmtrain(X,t,'method','SMO','kernel_function',nameKern,...
    'boxconstraint',C,'SMO_Opts',opts); 

else
    if strcmp(nameKern,'rbf')
        nameParam='rbf_sigma';
    end
    if strcmp(nameKern,'polynomial')
        nameParam='polyorder';
    end
    structTrain=svmtrain(X,t,'method','SMO','kernel_function',nameKern,...
        nameParam,param,'boxconstraint',C,'SMO_Opts',opts);

end

t_est=svmclassify(structTrain,XTest,'Showplot',false);