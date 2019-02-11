function [Accuracy,Gmean,Fmeasure]=Evaluate(Ttest,t_est,labels)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%   [Accuracy,Gmean,Fmeasure]=Evaluate(Ttest,t_est,labels)
% INPUTS:
%   Ttest: real test targets array
%   t_est: estimates test targets array
%   labels: array where first element is the minority target and second
%           element is the majority one.
% OUTPUTS:
%   Accuracy: accuracy percent  
%   Gmean: geometric mean percent
%   Fmeasure: f1-score percent
TP = sum((Ttest == labels(1)) & (t_est == labels(1)));
TF = sum((Ttest == labels(2)) & (t_est == labels(2)));
FP = sum((Ttest == labels(2)) & (t_est == labels(1)));
FN = sum((Ttest == labels(1)) & (t_est == labels(2)));

Acc_pos=TP/(TP + FN);
Acc_Neg=TF/(TF + FP);
Fmeasure = 2*TP/(2*TP + FN + FP)*100;
Gmean = sqrt(Acc_pos*Acc_Neg)*100;

Accuracy = mean(Ttest==t_est)*100;
