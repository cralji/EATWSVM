function t_new = OvO_ETWSVMClassify(model,Xtest)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%   t_new = OvO_ETWSVMClassify(model,Xtest)
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
c11 = model.param.c11;
c12 = model.param.c12;
nt = size(Xtest,1);
ij = model.targets;
Vote = zeros(nt,K);

for k = 1:nModels
    svmPlus = model.models{k,1};
    svmMinus = model.models{k,2};
    B1 = svmPlus.B;
    B2 = svmMinus.B;
    alpha = svmPlus.alpha;
    gamma = svmMinus.alpha;
%     S11 = svmPlus.S;
    S12 = svmPlus.Scross;
    S21 = svmMinus.Scross;
    
    Xc1 = svmPlus.X; nc1 = size(Xc1,1);
    Xc2 = svmMinus.X; nc2 = size(Xc2,1);
    ww1 = sqrt(svmPlus.ww);
    ww2 = sqrt(svmMinus.ww);
    Knew_1 = ComputeKern(Xtest,Xc1,kern) + ones(nt,nc1);
    Knew_2 = ComputeKern(Xtest,Xc2,kern) + ones(nt,nc2);
    
    f1 = (1/c11)*(Knew_2 - Knew_1*B1*S12)*alpha*(-1);
    f2 = (1/c12)*(Knew_1 - Knew_2*B2*S21)*gamma*(1);
    
    [~,ind] = min([abs(f1)./ww1 abs(f2)./ww2],[],2); % distance between both classes
    
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
