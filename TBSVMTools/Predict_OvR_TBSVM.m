function t_new = Predict_OvR_TBSVM(model,Xtest)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%   t_new = Predict_OvR_TBSVM(model,Xtest)
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
namekern = lower(kern.kernfunction);
nt = size(Xtest,1);
distance = zeros(nt,K);

for k = 1:K
    model_k = model.models{k};
    D = model_k.X;
    w1 = model_k.w1;
    b1 = model_k.b1;
    if any(strcmp(namekern,{'lin','linear'}))
        norm_w = model_k.norm_w1;
        distance(:,k) = (Xtest*w1 + b1*ones(nt,1))./norm_w; % distance matrix i-th samples and j-th class
        
    else
        den1 = model_k.norm_w1;
        K_x = ComputeKern(Xtest,D,kern);
        distance(:,k) = (K_x*w1 + b1*ones(nt,1))./den1; % distance matrix i-th samples and j-th class
    end
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


