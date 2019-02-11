function [BestPar,ti] = FindParamOvO_ETWSVM_fixedC(X,t,kern)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   BestPar=FindParamOvO_ETWSVM_fixedC(X,t,kern)
% INPUTS:
%   X: input samples matrix with which the parameters will be tuned.
%   t: targets array 
%   kern: struct kernel with fields:
%       .kernfunction: name kernel,  'lin': linearkernel
%                      and 'rbf': gaussian kernel. Default 'lin'
%       .param: scalar array with kernel's parameters
% OUTPUTS:
%   BestPar: exit struct with turned parameters. It has fields: 
%       .param: Struct with fields:
%               .c11 = 0.001, .c12=0.001, .c21=1 and .c22=1: These are 
%               regularization parameters.
%       .kern: kernel struct with best kernel function's parameter. 
%       .performance: Accuracy, geometric mean and Fmeasure measurements 
%                     for parameters set selected.  

%%
  np = 10;
  index = A_crossval_bal(t,np);

  %%  Training paratemeters
  
  if strcmp(lower(kern.kernfunction),'rbf')
    s0 = median(pdist(X));
    Ns = 10;
    Vs = linspace(0.1*s0,s0,Ns)';
  else
    Vs = 1;
    Ns = 1;
  end
  
  acm  = zeros(Ns,np);
  gmem = zeros(Ns,np);
  fmem = zeros(Ns,np);
  gmem_o = zeros(Ns,np);
  ti   = zeros(np,1);
  %% Loops
  param.c11 = 0.001;
  param.c12 = 0.001;
  param.c21 = 1;
  param.c22 = 1;
  for f=1:np
      %% Partition
      tic
      fprintf('\t %d \n',f)
      Xtrain = X(index~=f,:);
      Xtest = X(index==f,:);

      tTrain = t(index~=f);
      tTest = t(index==f);
      %% Loop
    for is = 1:Ns 
        kern.param = Vs(is);
        %% Training model
        svmStructs = OvO_ETWSVM(Xtrain,tTrain,param,kern);
        %% classification
        t_est = OvO_ETWSVMClassify(svmStructs,Xtest);
        %% evalute erformance
        [acm(is,f),fmem(is,f),~,gmem(is,f),~,gmem_o(is,f)] =...
          performance_MC(tTest,t_est,svmStructs.labels);
    end
      ti(f) = toc;
  end
  %% Best Parameters
  fme = squeeze(mean(fmem,2));
  gme_o = squeeze(mean(gmem_o,2));
  
  vfme = fme(:);
  vgme_o = gme_o(:);

  Fmax = max(vfme);
  index = find(vfme==Fmax);
  [~,ind]=max(vgme_o(index));

  [is]=ind2sub(Ns,index(ind));

  BestPar.param = param;
  
  kern.param = Vs(is);
  BestPar.kern = kern;
  
  
  BestPar.performance = {squeeze(acm(is,:)),squeeze(gmem(is,:)),...
    squeeze(gmem_o(is,:)),squeeze(fmem(is,:))}';