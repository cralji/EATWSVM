function [BestPar,ti]=FindParamETWSVM_fixedC(X,t,kern)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   BestPar=FindParamETWSVM_fixedC(X,t,kern)
% INPUTS:
%   X: input samples matrix with which the parameters will be tuned.
%   t: targets array 
%   kern: struct kernel with fields:
%       .kernfunction: name kernel,  'lin': linearkernel
%                      and 'rbf': gaussian kernel. Default 'lin'
%       .E: Transformated matrix of mahalanobis distance
%           R^{PxQ} Q<=P. Default indenty matrix with Q=P.
%       .param: scalar array with kernel's parameters
% OUTPUTS:
%   BestPar: exit struct with turned parameters. It has fields: 
%       .param: Struct with fields:
%               .c11 = 0.001, .c12=0.001, .c21=1 and .c22=1: These are 
%               regularization parameters.
%       .kern: the kernel struct with best parameter found.
%       .performance: Accuracy, geometric mean and Fmeasure measurements 
%                     for parameters set selected.  
  np=10;
  index=A_crossval_bal(t,np);
  Ns = 10;
  s0 = median(pdist(X));
  Vs =linspace(0.1*s0,s0,Ns);
  %%  Paramétros de entrenamiento
  exp = [-9 -7 -5 -3 -1 0 1 3 5 7 9];
  vC = 2.^exp;
  Nc = length(vC);
  ti = zeros(np,1);
  acm  = zeros(Ns,np);
  gmem = zeros(Ns,np);
  fmem = zeros(Ns,np);
  param.c11 = 0.001; 
  param.c12 = 0.001; 
  param.c21 = 1;     
  param.c22 = 1;  
%% loops
  for f=1:np
      %% Partition
      fprintf('\t %d \n',f)
      Xtrain = X(index~=f,:);
      Xtest = X(index==f,:);

      tTrain = t(index~=f);
      tTest = t(index==f);
      tic
      parfor is = 1:Ns
          %% Training model
          svmStrucs = ETWSVMTraining(Xtrain,tTrain,param,kern,Vs(is));
          %% classification
          [t_est]=ETWSVMClassify(svmStrucs,Xtest);
          %% evalute performance
          [acm(is,f),gmem(is,f),fmem(is,f)] =...
              Evaluate(tTest,t_est,svmStrucs.labels);
      end
      ti(f) = toc;
  end
  %% Best Parameters
%   ac = squeeze(mean(acm,2));
  gme = squeeze(mean(gmem,2));
  fme = squeeze(mean(fmem,2));

  vfme = fme(:);
  vgme = gme(:);
%   vac = ac(:);

  Fmax = max(vfme);
  index = find(vfme==Fmax);
  [~,ind]=max(vgme(index));

  [is]=ind2sub(Ns,index(ind));

     
  BestPar.param = param;
  kern.param = Vs(is);
  BestPar.kern = kern;
  
  BestPar.performance = {squeeze(acm(is,:)),squeeze(gmem(is,:)),...
    squeeze(fmem(is,:))}';