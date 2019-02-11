function t_new = Predict_OvR_SVM(model,Xtest)

K = model.K;
labels = model.labels;
kern = model.kern;
namekern = lower(kern.kernfunction);
nt = size(Xtest,1);
score = zeros(nt,K);

for k = 1:K
    model_k = model.models{k};
    score(:,k) = predict(model_k,Xtest);        
end

[~,index] = max(score,[],2);
Labels = repmat(string(labels'),nt,1);

ind = sub2ind([nt,K],(1:nt)',index);

t_new = Labels(ind);

if iscell(labels)
    t_new = cellstr(t_new);
else
    t_new = str2double(t_new);
end


