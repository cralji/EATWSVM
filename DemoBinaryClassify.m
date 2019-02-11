%% Demo of the differents algorithms
clear
close all 
clc
%% add paths
addpath(genpath('./utils')) % utils path
addpath(genpath('./ReSampling')) %resampling path
addpath(genpath('./MLmat-master')) % Metric learning path 
addpath(genpath('./SVMTools')) % SVM path
addpath(genpath('./TBSVMTools')) % TBSVM path
addpath(genpath('./WLTSVMTools')) % WLTSVM path
addpath(genpath('./ETWSVMTools')) % ETWSVM path
%% Load data
Data = load('./Iris.txt'); % GNU/Linux
% Data = load('.\Iris.txt'); % Windows
X = Data(:,1:end-1);
t = Data(:,end);
[~,~,labels] = SeparateClassesBinary(X,t);
NT = size(X,1);
indexTrain = randperm(NT,floor(0.7*NT));
indexTest = setdiff((1:NT),indexTrain);
Xtrain = X(indexTrain,:);
tTrain = t(indexTrain);
Xtest = X(indexTest,:);
tTest = t(indexTest);
%% 
acc= zeros(11,1);
gm = zeros(11,1);
fm = zeros(11,1);
modelsname = {'SVM','WSVM','SVM-SMOTE','TBSVM','WLTSVM','ETWSVM','ETWSVM-CKA',...
    'ETWSVM-CKA-SMOTE','ETWSVM_fixedC','ETWSVM-CKA_fixedC','ETWSVM-CKA-SMOTE_fixedC'}';
%% SVM Training
kern.function = 'rbf';
BestParam = FindParamSVM(Xtrain,tTrain,kern); % search best parameters
kern = BestParam.kern; % best parameters to train 
t_est = SVMTraining(Xtrain,tTrain,Xtest,kern);
[acc(1),gm(1),fm(1)] = Evaluate(tTest,t_est,labels);
fprintf('---------------------------------------------------\n')
fprintf('SVM \t ACC=%.4f \t GM=%.4f \t FM=%.4f\n',acc(1),gm(1),fm(1))
fprintf('---------------------------------------------------\n')

%% WSVM Training
kern.function = 'rbf';
BestParam = FindParamWSVM(Xtrain,tTrain,kern); % search best parameters
kern = BestParam.kern; % best parameters to train 
t_est = WSVMTraining(Xtrain,tTrain,Xtest,kern);
[acc(2),gm(2),fm(2)] = Evaluate(tTest,t_est,labels);
fprintf('-------------------------------------------------------\n')
fprintf('WSVM \t ACC=%.4f \t GM=%.4f \t FM=%.4f\n',acc(2),gm(2),fm(2))
fprintf('-------------------------------------------------------\n')

%% SMOTE-SVM training
kern.function = 'rbf';
BestParam = FindParamSVM_SMOTE(Xtrain,tTrain,kern); % search best parameters
kern = BestParam.kern; % best parameters to train 
v = BestParam.k;
[X1,X2,labels] = SeparateClassesBinary(X,t);
nmin=size(X1,1);
nmax=size(X2,1);
N=((nmax-nmin)/nmin)*100;
Synthetic = SMOTE(X1,N,v);
Xos = [Xtrain;Synthetic];
tos=[tTrain;repmat(labels(1),size(Synthetic,1),1)];
t_est = SVMTraining(Xos,tos,Xtest,kern);
[acc(3),gm(3),fm(3)] = Evaluate(tTest,t_est,labels);
fprintf('---------------------------------------------------------------\n')
fprintf('SMOTE-SVM \t ACC=%.4f \t GM=%.4f \t FM=%.4f\n',acc(3),gm(3),fm(3))
fprintf('---------------------------------------------------------------\n')
%% TBSVM training
kern.kernfunction = 'rbf';
BestParam = FindParamTBSVM(X,t,kern);
param = BestParam.param; % best regularization parameteres set 
kern = BestParam.kern; % best parameters to train 
model = TBSVMTraining(Xtrain,tTrain,kern,param); % Train model 
t_est = TBSVMClassify(Xtest,model); % Classify Ttest
[acc(4),gm(4),fm(4)] = Evaluate(tTest,t_est,labels);
fprintf('---------------------------------------------------------------\n')
fprintf('TBSVM \t ACC=%.4f \t GM=%.4f \t FM=%.4f\n',acc(4),gm(4),fm(4))
fprintf('---------------------------------------------------------------\n')
%% WLTSVM Training
tipokern = 'rbf';
FunPara.kerfPara.type = tipokern;
BestParam = FindParamWLTSVM(Xtrain,tTrain,tipokern);
[Xc1,Xc2,labels]=SeparateClassesBinary(Xtrain,tTrain);
DataTrain.A=Xc1;
DataTrain.B=Xc2;
FunPara.p1=BestParam.c1;
FunPara.p2=BestParam.c2;
FunPara.kerfPara.pars= BestParam.param;
[label_new]=WLTSVM(Xtest,DataTrain,FunPara); % out= Accu Gmean Fmeasure
[acc(5),gm(5),fm(5)] = Evaluate(tTest,label_new,labels);
fprintf('---------------------------------------------------------------\n')
fprintf('WLTSVM \t ACC=%.4f \t GM=%.4f \t FM=%.4f\n',acc(5),gm(5),fm(5))
fprintf('---------------------------------------------------------------\n')
%% ETWSVM training
kern.kernfunction = 'rbf';
BestParam = FindParamETWSVM(Xtrain,tTrain,kern);
param = BestParam.param; % best regularization parameteres set 
kern = BestParam.kern; % best parameters to train 
model = ETWSVMTraining(Xtrain,tTrain,param,kern); % Train model 
t_est = ETWSVMClassify(model,Xtest); % Classify Ttest
[acc(6),gm(6),fm(6)] = Evaluate(tTest,t_est,labels);
fprintf('---------------------------------------------------------------\n')
fprintf('ETWSVM \t ACC=%.4f \t GM=%.4f \t FM=%.4f\n',acc(6),gm(6),fm(6))
fprintf('---------------------------------------------------------------\n')
%% ETWSVM CKA
kern.kernfunction = 'rbf';
BestParam = FindParamETWSVM_CKA(Xtrain,tTrain,kern);
%%Metric Learning -- CKA,
opts = ckaoptimset();
opts.showCommandLine=false;
opts.showWindow=false;
opts.maxiter = 120;
opts.Q = 0.95;
[Xus,tus] = US_vnn(Xtrain,tTrain,8);
[~,~,labels] = SeparateClassesBinary(Xtrain,tTrain);
nc1 = size(Xus(tus==labels(1)),1);
nc2 = size(Xus(tus==labels(2)),1);
l = [ones(nc1,1);-1*ones(nc2,1)];
[Xc1,Xc2,~] = SeparateClassesBinary(Xus,tus);
L = double(bsxfun(@eq,l,l'));
E = kMetricLearningMahalanobis([Xc1;Xc2],L,l,opts);
%%end metric learning
kern.E = E;
param = BestParam.param; % best regularization parameteres set 
model = ETWSVMTraining(Xtrain,tTrain,param,kern); % Train model 
t_est = ETWSVMClassify(model,Xtest); % Classify Ttest
[acc(7),gm(7),fm(7)] = Evaluate(tTest,t_est,labels);
fprintf('---------------------------------------------------------------\n')
fprintf('ETWSVM-CKA \t ACC=%.4f \t GM=%.4f \t FM=%.4f\n',[acc(7),gm(7),fm(7)]')
fprintf('---------------------------------------------------------------\n')
%% ETWSVM-CKA-SMOTE
kern.kernfunction = 'rbf';
BestParam = FindParamETWSVM_CKA_SMOTE(Xtrain,tTrain,kern);
%%Metric Learning -- CKA,
opts = ckaoptimset();
opts.showCommandLine=false;
opts.showWindow=false;
opts.maxiter = 120;
opts.Q = 0.95;
[Xus,tus] = US_vnn(Xtrain,tTrain,8);
[~,~,labels] = SeparateClassesBinary(Xtrain,tTrain);
nc1 = size(Xus(tus==labels(1)),1);
nc2 = size(Xus(tus==labels(2)),1);
l = [ones(nc1,1);-1*ones(nc2,1)];
[Xc1,Xc2,~] = SeparateClassesBinary(Xus,tus);
L = double(bsxfun(@eq,l,l'));
E = kMetricLearningMahalanobis([Xc1;Xc2],L,l,opts);
%%end metric learning
kern.E = E;
param = BestParam.param; % best regularization parameteres set 
v = BestParam.k;
[X1,X2,~] = SeparateClassesBinary(X,t);
nmin=size(X1,1);
nmax=size(X2,1);
N=((nmax-nmin)/nmin)*100;
Synthetic = SMOTE(X1,N,v);
Xos = [Xtrain;Synthetic];
tos=[tTrain;repmat(labels(1),size(Synthetic,1),1)];
model = ETWSVMTraining(Xos,tos,param,kern); % Train model 
t_est = ETWSVMClassify(model,Xtest); % Classify Ttest
[acc(8),gm(8),fm(8)] = Evaluate(tTest,t_est,labels);
fprintf('---------------------------------------------------------------\n')
fprintf('ETWSVM-CKA-SMOTE \t ACC=%.4f \t GM=%.4f \t FM=%.4f\n',[acc(8),gm(8),fm(8)]')
fprintf('---------------------------------------------------------------\n')

%% ETWSVM training fixed regularization parameters
kern.kernfunction = 'rbf';
BestParam = FindParamETWSVM_fixedC(Xtrain,tTrain,kern);
param = BestParam.param; % best regularization parameteres set 
kern = BestParam.kern; % best parameters to train 
model = ETWSVMTraining(Xtrain,tTrain,param,kern); % Train model 
t_est = ETWSVMClassify(model,Xtest); % Classify Ttest
[acc(9),gm(9),fm(9)] = Evaluate(tTest,t_est,labels);
fprintf('---------------------------------------------------------------\n')
fprintf('ETWSVM \t ACC=%.4f \t GM=%.4f \t FM=%.4f\n',acc(9),gm(9),fm(9))
fprintf('---------------------------------------------------------------\n')
%% ETWSVM CKA
kern.kernfunction = 'rbf';
%%Metric Learning -- CKA,
opts = ckaoptimset();
opts.showCommandLine=false;
opts.showWindow=false;
opts.maxiter = 120;
opts.Q = 0.95;
[Xus,tus] = US_vnn(Xtrain,tTrain,8);
[~,~,labels] = SeparateClassesBinary(Xtrain,tTrain);
nc1 = size(Xus(tus==labels(1)),1);
nc2 = size(Xus(tus==labels(2)),1);
l = [ones(nc1,1);-1*ones(nc2,1)];
[Xc1,Xc2,~] = SeparateClassesBinary(Xus,tus);
L = double(bsxfun(@eq,l,l'));
E = kMetricLearningMahalanobis([Xc1;Xc2],L,l,opts);
%%end metric learning
kern.E = E;
param.c11 = 0.001; param.c12 = 0.001;
param.c21 = 1; param.c22 = 1;
model = ETWSVMTraining(Xtrain,tTrain,param,kern); % Train model 
t_est = ETWSVMClassify(model,Xtest); % Classify Ttest
[acc(10),gm(10),fm(10)] = Evaluate(tTest,t_est,labels);
fprintf('---------------------------------------------------------------\n')
fprintf('ETWSVM-CKA \t ACC=%.4f \t GM=%.4f \t FM=%.4f\n',[acc(10),gm(10),fm(10)]')
fprintf('---------------------------------------------------------------\n')
%% ETWSVM-CKA-SMOTE
kern.kernfunction = 'rbf';
BestParam = FindParamETWSVM_CKA_SMOTE_fixedC(Xtrain,tTrain,kern);
%%Metric Learning -- CKA,
opts = ckaoptimset();
opts.showCommandLine=false;
opts.showWindow=false;
opts.maxiter = 120;
opts.Q = 0.95;
[Xus,tus] = US_vnn(Xtrain,tTrain,8);
[~,~,labels] = SeparateClassesBinary(Xtrain,tTrain);
nc1 = size(Xus(tus==labels(1)),1);
nc2 = size(Xus(tus==labels(2)),1);
l = [ones(nc1,1);-1*ones(nc2,1)];
[Xc1,Xc2,~] = SeparateClassesBinary(Xus,tus);
L = double(bsxfun(@eq,l,l'));
E = kMetricLearningMahalanobis([Xc1;Xc2],L,l,opts);
%%end metric learning
kern.E = E;
param = BestParam.param; % best regularization parameteres set 
v = BestParam.k;
[X1,X2,~] = SeparateClassesBinary(X,t);
nmin=size(X1,1);
nmax=size(X2,1);
N=((nmax-nmin)/nmin)*100;
Synthetic = SMOTE(X1,N,v);
Xos = [Xtrain;Synthetic];
tos=[tTrain;repmat(labels(1),size(Synthetic,1),1)];
model = ETWSVMTraining(Xos,tos,param,kern); % Train model 
t_est = ETWSVMClassify(model,Xtest); % Classify Ttest
[acc(11),gm(11),fm(11)] = Evaluate(tTest,t_est,labels);
fprintf('---------------------------------------------------------------\n')
fprintf('ETWSVM-CKA-SMOTE \t ACC=%.4f \t GM=%.4f \t FM=%.4f\n',[acc(11),gm(11),fm(11)]')
fprintf('---------------------------------------------------------------\n')
%% 
T = table(modelsname,acc,gm,fm);
T




