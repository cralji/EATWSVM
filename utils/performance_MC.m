function [acc,fscore_mu,fscore_M,Gmean_mu,Gmean_M,Gmean_our,TP,TN,FP,FN] =...
    performance_MC(t,te,labels)

K = length(labels); % number of classes
TP = zeros(K,1);
FP = zeros(K,1);
TN = zeros(K,1);
FN = zeros(K,1);

for k = 1:K
    index1 = string(t)==string(labels(k));
    index2 = string(te)==string(labels(k));
    
    TP(k) = sum(index1==1 & index2==1);
    TN(k) = sum(index2==0 & index1==0);
    
    FP(k) = sum(index2==1 & index1==0);
    FN(k) = sum(index2==0 & index1==1);
end

acc = sum((TP + TN)./(TP + TN + FP + FN))/K*100;

fscore_mu = 2*sum(TP)/(sum(TP + FN)+sum(TP + FP))*100;


temp = TP./(TP + FP);
index = isnan(temp);
temp(index) = 0;

precision_M = mean(temp);
recall = mean(TP./(TP + FN));
fscore_M = 2*precision_M*recall/(precision_M + recall)*100;

Gmean_M = mean(sqrt(TP./(TP+FN).*TN./(TN+FP)))*100;
Gmean_mu = sqrt(sum(TP)/sum(TP+FN)*sum(TN)/sum(TN+FP))*100;

Gmean_our = sqrt(prod(TP./(TP+FN)))*100;
