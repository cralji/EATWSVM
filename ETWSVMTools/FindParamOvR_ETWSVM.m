function [BestPar,ti] = FindParamOvR_ETWSVM(X,t,kern)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   BestPar = FindParamOvR_ETWSVM(X,t,kern)
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
%       .c11,.c12,.c21,.c22: The best regularization parameters set by
%                            F-measure and geometric mean.
%       .kern: kernel struct with best paremeter
%       .performance: performance measurements 
%                     for parameters set selected.  

  %% Buscando el parámetro del kernel gaussiano
  np = 10;
  index = A_crossval_bal(t,np);

  %%  Paramétros de entrenamiento
  exp = [-9 -7 -5 -3 -1 0 1 3 5 7 9];
  vC = 2.^exp;
  Nc = length(vC);
  if strcmp(lower(kern.kernfunction),'rbf')
    s0 = median(pdist(X));
    Ns = 10;
    Vs = linspace(0.1*s0,s0,Ns)';
  else
    Vs = 1;
    Ns = 1;
  end
  
  acm  = zeros(Ns,Nc,Nc,np);
  gmem = zeros(Ns,Nc,Nc,np);
  fmem = zeros(Ns,Nc,Nc,np);
  gmem_o = zeros(Ns,Nc,Nc,np);
  ti   = zeros(np,1);
  %% Loops
%   param.c11 = 0.001;
%   param.c12 = 0.001;
%   param.c21 = 1;
%   param.c22 = 1;
  for f=1:np
      %% Partition
      tic
      fprintf('\t %d \n',f)
      Xtrain = X(index~=f,:);
      Xtest = X(index==f,:);

      tTrain = t(index~=f);
      tTest = t(index==f);
      %% Loop
%       param.c2 = 1;
    for is =1:Ns
        kern.param = Vs(is);
      for ic1=1:Nc
          param.c11 = vC(ic1);
          param.c12 = vC(ic1);
          parfor ic2=1:Nc
              %% Training model
              svmStructs = OvR_ETWSVM(Xtrain,tTrain,param,kern,[],[],[],...
                  vC(ic2),vC(ic2));
              %% classification
              t_est = OvR_ETWSVMClassify(svmStructs,Xtest);
              %% evalute performance
              [acm(is,ic1,ic2,f),fmem(is,ic1,ic2,f),~,gmem(is,ic1,ic2,f),~,gmem_o(is,ic1,ic2,f)] =...
                  performance_MC(tTest,t_est,svmStructs.labels);
          end
      end
    end
      ti(f) = toc;
  end
  %% Best Parameters
  fme = squeeze(mean(fmem,4));
  gme_o = squeeze(mean(gmem_o,4));
  
  vfme = fme(:);
  vgme_o = gme_o(:);


  Fmax = max(vfme);
  index = find(vfme==Fmax);
  [~,ind]=max(vgme_o(index));

  [is,ic1,ic2]=ind2sub([Ns,Nc,Nc],index(ind));

  param.c11 = vC(ic1);
  param.c12 = vC(ic1);
  param.c21 = vC(ic2);
  param.c22 = vC(ic2);
  BestPar.param = param;
  kern.param = Vs(is);
  BestPar.kern = kern;
  
  BestPar.performance = {squeeze(acm(is,ic1,ic2,:)),squeeze(gmem(is,ic1,ic2,:)),...
    squeeze(gmem_o(is,ic1,ic2,:)),squeeze(fmem(is,ic1,ic2,:))}';