%% Demo of the differents algorithms
clear
close all 
clc
%% add paths
addpath(genpath('./utils')) % utils path
addpath(genpath('./ReSampling')) %resampling path
addpath(genpath('./MLmat-master')) % Metric learning path (necessary to CKA)
addpath(genpath('./SVMTools')) % SVM path
addpath(genpath('./TBSVMTools')) % TBSVM path
addpath(genpath('./WLTSVMTools')) % WLTSVM path
addpath(genpath('./ETWSVMTools')) % ETWSVM path
%% Load data
data=readtable('iris.dat');  %The last column have targets (t).
X = table2array(data(:,1:end-1));
if isscalar(data(1,end))
    t = table2array(data(:,end));
else
    t = table2cell(data(:,end));
end
NT = size(X,1);
index = A_crossval_bal(t,5);
Xtrain = X(index==1,:);
tTrain = t(index==1);
Xtest = X(index~=1,:);
tTest = t(index~=1);
%%
acc= zeros(12,1);
fmem = zeros(12,1);
modelsname = {'OvR-SVM';'OvO-SVM';'OvR-TBSVM';'OvO-TBSVM';'OvR-ETWSVM';'OvO-ETWSVM';...
    'OvR-ETWSVM-CKA';'OvO-ETWSVM-CKA';'OvR-ETWSVM_fixedC';'OvO-ETWSVM_fixedC';...
    'OvR-ETWSVM-CKA_fixedC';'OvO-ETWSVM-CKA_fixedC'};
%% OvR-SVM
kern.kernfunction = 'rbf';
BestParam = FindParamOvR_SVM(Xtrain,tTrain,kern);
param = BestParam.param;
kern = BestParam.kern;
models = OvR_SVM(Xtrain,tTrain,param,kern); % Training model
t_est = Predict_OvR_SVM(models,Xtest); % classification
[acc(1),fmem(1)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvR-SVM \t ACC=%.4f  \t FM_mu=%.4f\n',acc(1),fmem(1))
fprintf('---------------------------------------------------\n')
%% OvO-SVM
kern.kernfunction = 'rbf';
BestParam = FindParamOvO_SVM(Xtrain,tTrain,kern);
param = BestParam.param;
kern = BestParam.kern;
models = OvO_SVM(Xtrain,tTrain,param,kern); % Training model
t_est = Predict_OvO_SVM(models,Xtest); % classification
[acc(2),fmem(2)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvO-SVM \t ACC=%.4f  \t FM_mu=%.4f\n',acc(2),fmem(2))
fprintf('---------------------------------------------------\n')
%% OvR-TBSVM
kern.kernfunction = 'rbf';
BestParam = FindParamOvR_TBSVM(Xtrain,tTrain,kern);
param = BestParam.param;
kern = BestParam.kern;
models = OvR_TBSVM(Xtrain,tTrain,param,kern); % Training model
t_est = Predict_OvR_TBSVM(models,Xtest); % classification
[acc(3),fmem(3)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvR-TBSVM \t ACC=%.4f  \t FM_mu=%.4f\n',acc(3),fmem(3))
fprintf('---------------------------------------------------\n')
%% OvO-TBSVM
kern.kernfunction = 'rbf';
BestParam = FindParamOvO_TBSVM(Xtrain,tTrain,kern);
param = BestParam.param;
kern = BestParam.kern;
models = OvO_TBSVM(Xtrain,tTrain,param,kern); % Training model
t_est = Predict_OvO_TBSVM(models,Xtest); % classification
[acc(4),fmem(4)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvO-TBSVM \t ACC=%.4f  \t FM_mu=%.4f\n',acc(4),fmem(4))
fprintf('---------------------------------------------------\n')
%% OvR-ETWSVM
kern.kernfunction = 'rbf';
BestParam = FindParamOvR_ETWSVM(Xtrain,tTrain,kern);
param = BestParam.param;
kern = BestParam.kern;
models = OvR_ETWSVM(Xtrain,tTrain,param,kern); % Training model
t_est = OvR_ETWSVMClassify(models,Xtest); % classification
[acc(5),fmem(5)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvR-ETWSVM \t ACC=%.4f  \t FM_mu=%.4f\n',acc(5),fmem(5))
fprintf('---------------------------------------------------\n')
%% OvO-ETWSVM
kern.kernfunction = 'rbf';
BestParam = FindParamOvO_ETWSVM(Xtrain,tTrain,kern);
param = BestParam.param;
kern = BestParam.kern;
models = OvO_ETWSVM(Xtrain,tTrain,param,kern); % Training model
t_est = OvO_ETWSVMClassify(models,Xtest); % classification
[acc(6),fmem(6)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvO-ETWSVM \t ACC=%.4f  \t FM_mu=%.4f\n',acc(6),fmem(6))
fprintf('---------------------------------------------------\n')
%% OvR-ETWSVM-CKA
kern.kernfunction = 'rbf';
BestParam = FindParamOvR_ETWSVM_CKA(Xtrain,tTrain,kern);
param = BestParam.param;
%%metric learning CKA
opts = ckaoptimset();
opts.showCommandLine=false;
opts.showWindow=false;
opts.maxiter = 120;
opts.Q = 0.95;
[Xsort,l,labels] = SeparateClassesMulticlass(Xtrain,tTrain,0);
L = double(bsxfun(@eq,l,l'));
E = kMetricLearningMahalanobis(Xsort,L,l,opts);
%%END CKA
kern.E = E;
models = OvR_ETWSVM(Xtrain,tTrain,param,kern); % Training model
t_est = OvR_ETWSVMClassify(models,Xtest); % classification
[acc(7),fmem(7)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvR-ETWSVM-CKA \t ACC=%.4f  \t FM_mu=%.4f\n',acc(7),fmem(7))
fprintf('---------------------------------------------------\n')
%% OvO-ETWSVM-CKA 
kern.kernfunction = 'rbf';
BestParam = FindParamOvO_ETWSVM_CKA(Xtrain,tTrain,kern);
param = BestParam.param;
%%metric learning CKA
opts = ckaoptimset();
opts.showCommandLine=false;
opts.showWindow=false;
opts.maxiter = 120;
opts.Q = 0.95;
[Xsort,l,labels] = SeparateClassesMulticlass(Xtrain,tTrain,0);
L = double(bsxfun(@eq,l,l'));
E = kMetricLearningMahalanobis(Xsort,L,l,opts);
%%END CKA
kern.E = E;
models = OvO_ETWSVM(Xtrain,tTrain,param,kern); % Training model
t_est = OvO_ETWSVMClassify(models,Xtest); % classification
[acc(8),fmem(8)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvO-ETWSVM-CKA \t ACC=%.4f  \t FM_mu=%.4f\n',acc(8),fmem(8))
fprintf('---------------------------------------------------\n')
%% OvR-ETWSVM; fixed c11=c12=0.001 and c21=c22=1
kern.kernfunction = 'rbf';
BestParam = FindParamOvR_ETWSVM_fixedC(Xtrain,tTrain,kern);
param = BestParam.param;
kern = BestParam.kern;
models = OvR_ETWSVM(Xtrain,tTrain,param,kern); % Training model
t_est = OvR_ETWSVMClassify(models,Xtest); % classification
[acc(9),fmem(9)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvR-ETWSVM_fixedC \t ACC=%.4f  \t FM_mu=%.4f\n',acc(9),fmem(9))
fprintf('---------------------------------------------------\n')
%% OvO-ETWSVV fixed c11=c12=0.001 and c21=c22=1
kern.kernfunction = 'rbf';
BestParam = FindParamOvO_ETWSVM_fixedC(Xtrain,tTrain,kern);
param = BestParam.param;
kern = BestParam.kern;
models = OvO_ETWSVM(Xtrain,tTrain,param,kern); % Training model
t_est = OvO_ETWSVMClassify(models,Xtest); % classification
[acc(10),fmem(10)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvO-ETWSVM_fixedC \t ACC=%.4f  \t FM_mu=%.4f\n',acc(10),fmem(10))
fprintf('---------------------------------------------------\n')
%% OvR-ETWSVM fixed c11=c12=0.001 and c21=c22=1
kern.kernfunction = 'rbf';
param.c11 = 0.001;
param.c12 = 0.001;
param.c21 = 1;
param.c22 = 1;
%%metric learning CKA
opts = ckaoptimset();
opts.showCommandLine=false;
opts.showWindow=false;
opts.maxiter = 120;
opts.Q = 0.95;
[Xsort,l,labels] = SeparateClassesMulticlass(Xtrain,tTrain,0);
L = double(bsxfun(@eq,l,l'));
E = kMetricLearningMahalanobis(Xsort,L,l,opts);
%%END CKA
kern.E = E;
models = OvR_ETWSVM(Xtrain,tTrain,param,kern); % Training model
t_est = OvR_ETWSVMClassify(models,Xtest); % classification
[acc(11),fmem(11)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvR-ETWSVM-CKA_fixedC \t ACC=%.4f  \t FM_mu=%.4f\n',acc(11),fmem(11))
fprintf('---------------------------------------------------\n')
%% OvO-ETWSVM fixed c11=c12=0.001 and c21=c22=1
kern.kernfunction = 'rbf';
param.c11 = 0.001;
param.c12 = 0.001;
param.c21 = 1;
param.c22 = 1;
%%metric learning CKA
opts = ckaoptimset();
opts.showCommandLine=false;
opts.showWindow=false;
opts.maxiter = 120;
opts.Q = 0.95;
[Xsort,l,labels] = SeparateClassesMulticlass(Xtrain,tTrain,0);
L = double(bsxfun(@eq,l,l'));
E = kMetricLearningMahalanobis(Xsort,L,l,opts);
%%END CKA
kern.E = E;
models = OvO_ETWSVM(Xtrain,tTrain,param,kern); % Training model
t_est = OvO_ETWSVMClassify(models,Xtest); % classification
[acc(12),fmem(12)] = performance_MC(tTest,t_est,models.labels);
fprintf('---------------------------------------------------\n')
fprintf('OvO-ETWSVM-CKA_fixedC \t ACC=%.4f  \t FM_mu=%.4f\n',acc(12),fmem(12))
fprintf('---------------------------------------------------\n')
%% 
T = table(modelsname,acc,fmem);
T