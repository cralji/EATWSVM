function t_new = Predict_OvO_SVM(model,Xtest)

nModels = model.nModels;
K = model.K;
labels = model.labels;

nt = size(Xtest,1);
ij = model.targets;
Vote = zeros(nt,K);

for k = 1:nModels
    ind = zeros(nt,1);
    model_k = model.models{k};
    est = predict(model_k,Xtest);
    ind(est==1) = 1;  ind(est==-1) = 2;    
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
