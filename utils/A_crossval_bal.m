function [indices] = A_crossval_bal(label,np)

%Balanced cross validation partition - kfold approach
%label: input labels
%np: number of partitions or percentage of training set

%% modificado por Cristian Jimenez
if iscell(label)
    label_temp = zeros(length(label),1);
    targets = unique(label);
    nC = length(targets);
    strLabel = string(label);
    for k = 1:nC
        label_temp(strLabel==string(targets(k))) = k;
    end
    clear label strLabel targets
    label = label_temp;
    clear label_temp
end
%% fin modificación

N = numel(label);

if np < N
    
    nC = unique(label);
    indices = zeros(numel(label),1);
    for i = 1 : numel(nC)
        indi = find(label==nC(i));
        [indic] = crossvalind('Kfold',length(indi),np);
        indices(indi)=indic;
        
    end
    
else
    indices = 1 : N;
    indices = indices';
end