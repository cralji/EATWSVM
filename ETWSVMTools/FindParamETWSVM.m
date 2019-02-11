function [BestPar,ti]=FindParamETWSVM(X,t,kern)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   BestPar=FindParamETWSVM_CKA(X,t,kern)
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
%       .param: Struct with best regularization parameters set by
%               F-measure and geometric mean. It has fields
%               .c11, .c12, .c21 and .c22: These are regularization parameters.
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
  acm  = zeros(Nc,Nc,Ns,np);
  gmem = zeros(Nc,Nc,Ns,np);
  fmem = zeros(Nc,Nc,Ns,np);
  %% Loops
  for f=1:np
      %% Partition
      fprintf('\t %d \n',f)
      Xtrain = X(index~=f,:);
      Xtest = X(index==f,:);

      tTrain = t(index~=f);
      tTest = t(index==f);
      tic
      for is = 1:Ns
          kern.param = Vs(is);
          for ic1=1:Nc
              param.c11 = vC(ic1);
              param.c12 = vC(ic1);
              parfor ic2=1:Nc
                  %% Training model
                  svmStrucs = ETWSVMTraining(Xtrain,tTrain,param,kern,[],[],[],vC(ic2),vC(ic2));
                  %% classification
                  [t_est]=ETWSVMClassify(svmStrucs,Xtest);
                  %% evalute performance
                  [acm(ic1,ic2,is,f),gmem(ic1,ic2,is,f),fmem(ic1,ic2,is,f)] =...
                      Evaluate(tTest,t_est,svmStrucs.labels);
              end
          end
      end
      ti(f) = toc;
  end
  %% Best Parameters
%   ac = squeeze(mean(acm,4));
  gme = squeeze(mean(gmem,4));
  fme = squeeze(mean(fmem,4));

  vfme = fme(:);
  vgme = gme(:);
%   vac = ac(:);

  Fmax = max(vfme);
  index = find(vfme==Fmax);
  [~,ind]=max(vgme(index));

  [ic1,ic2,is]=ind2sub([Nc,Nc,Ns],index(ind));

  param.c11 = vC(ic1);
  param.c12 = vC(ic1);
  param.c21 = vC(ic2);
  param.c22 = vC(ic2);
  BestPar.param = param;
  kern.param = Vs(is);
  BestPar.kern = kern;
  
  BestPar.performance = {squeeze(acm(ic1,ic2,is,:)),squeeze(gmem(ic1,ic2,is,:)),...
    squeeze(fmem(ic1,ic2,is,:))}';
% delete(gcp('nocreate'))