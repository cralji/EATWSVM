function t_new = Predict_OvO_TBSVM(model,Xtest)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%   t_new = Predict_OvO_TBSVM(model,Xtest)
% INPUTS:
%   model: struct obtained of to train with OvR_ETWSVM.m. It has fields:
%       .models: cell matrox with structs for each binary model training
%       .param: struct with regularization parameters. 
%       .kern: struct kernel
%       .labels: array with name targets
%       .K: number of classes
%       .nModels: number of the total models K(K-1)/2
%       .targets: cell array with the differents combitations between
%                 classes
%   Xtest: test sample matrix, sample for row.
% OUTPUTS:
%   t_new: estimate targets array
nModels = model.nModels;
K = model.K;
labels = model.labels;
kern = model.kern;
namekern = lower(kern.kernfunction);

nt = size(Xtest,1);
ij = model.targets;
Vote = zeros(nt,K);

for k = 1:nModels
    model_k = model.models(k,:);
    D1 = model_k{1}.X;
    D2 = model_k{2}.X;
    w1 = model_k{1}.w1;
    w2 = model_k{2}.w1;
    
    b1 = model_k{1}.b1;
    b2 = model_k{2}.b1;
    
    if any(strcmp(namekern,{'lin','linear'}))
        norm_w1 = model_k{1}.norm_w1;
        norm_w2 = model_k{2}.norm_w1;
        f1 = (Xtest*w1 + b1*ones(nt,1))./norm_w1;
        f2 = (Xtest*w2 + b2*ones(nt,1))./norm_w2;% distance matrix i-th samples and j-th class
    else
        den1 = model_k{1}.norm_w1;
        den2 = model_k{2}.norm_w1;
        K_x = ComputeKern(Xtest,D1,kern);
        f1 = (K_x*w1 + b1*ones(nt,1))./den1;
        K_x = ComputeKern(Xtest,D2,kern);
        f2 = (K_x*w2 + b2*ones(nt,1))./den2;% distance matrix i-th samples and j-th class
    end
    
    [~,ind] = min([abs(f1) abs(f2)],[],2); % distance between both classes
    
    class = ij{k}(ind);
    ind = sub2ind([nt,K],(1:nt)',class');
    Vote(ind) = Vote(ind) + 1; % distance matrix i-th samples and j-th class
end

[~,index] = max(Vote,[],2);
Labels = repmat(string(labels'),nt,1);

ind = sub2ind([nt,nModels],(1:nt)',index);

t_new = Labels(ind);

if iscell(labels)
    t_new = cellstr(t_new);
else
    t_new = str2double(t_new);
end
