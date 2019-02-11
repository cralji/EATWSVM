function t_new = OvR_ETWSVMClassify(model,Xtest)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%   t_new = OvR_ETWSVMClassify(model,Xtest)
% INPUTS:
%   model: struct obtained of to train with OvR_ETWSVM.m. It has fields:
%       .models: cell array with structs for each binary model training
%       .param: struct with regularization parameters. 
%       .kern: struct kernel
%       .labels: array with name targets
%       .K: number of classes
%   Xtest: test sample matrix, sample for row.
% OUTPUTS:
%   t_new: estimate targets array


K = model.K;
labels = model.labels;
kern = model.kern;
c1 = model.param.c11;
nt = size(Xtest,1);

distance = zeros(nt,K);

for k = 1:K
    svmPlus = model.models{k};
    B1 = svmPlus.B;
    alpha = svmPlus.alpha;
%     S11 = svmPlus.S;
    S12 = svmPlus.Scross;
    Xc1 = svmPlus.X; nc1 = size(Xc1,1);
    Xc2 = svmPlus.X2; nc2 = size(Xc2,1);
    ww1 = sqrt(svmPlus.ww);
    Knew_1 = ComputeKern(Xtest,Xc1,kern) + ones(nt,nc1);
    Knew_2 = ComputeKern(Xtest,Xc2,kern) + ones(nt,nc2);
    f1 = (1/c1)*(Knew_2 - Knew_1*B1*S12)*alpha*(-1);
    distance(:,k) = abs(f1)./ww1; % distance matrix i-th samples and j-th class
end

[~,index] = min(distance,[],2);
Labels = repmat(string(labels'),nt,1);

ind = sub2ind([nt,K],(1:nt)',index);

t_new = Labels(ind);

if iscell(labels)
    t_new = cellstr(t_new);
else
    t_new = str2double(t_new);
end


