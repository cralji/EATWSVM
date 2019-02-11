function [BestPar,ti] = FindParamOvR_TBSVM(X,t,kern)
  %@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   BestPar = FindParamOvR_TBSVM(X,t,kern)
% INPUTS:
%   X: input samples matrix with which the parameters will be tuned.
%   t: targets array 
%   kern: struct kernel with fields:
%       .kernfunction: name kernel,  'lin': linearkernel
%                      and 'rbf': gaussian kernel. Default 'lin'
%       .param: scalar array with kernel's parameters
% OUTPUTS:
%   BestPar: exit struct with turned parameters. It has fields: 
%       .param: Struct with best regularization parameters set by
%               F-measure and geometric mean. It has fields
%               .c11, .c12, .c21 and .c22: These are regularization parameters.
%       .kern: the kernel struct with best parameter found.
%       .performance: Accuracy, geometric mean and Fmeasure measurements 
%                     for parameters set selected.  
  np = 10;
  index = A_crossval_bal(t,np);
 warning('off','all');
  %%  Paramétros de entrenamiento
  exp = [-9 -7 -5 -3 -1 0 1 3 5 7 9];
  vC = 2.^exp;
  Nc = length(vC);
  namekern = lower(kern.kernfunction);
  Ns = 10;
  if strcmp(namekern,'rbf')
        so = median(pdist(X));
        vS = linspace(0.01*so,so,Ns);
  elseif any(strcmp(namekern,{'lin','linear'}))
        vS = 1;
        Ns = 1;
  end
  acm  = zeros(Nc,Nc,Ns,np);
  gmem = zeros(Nc,Nc,Ns,np);
  fmem = zeros(Nc,Nc,Ns,np);
  gmem_o = zeros(Nc,Nc,Ns,np);
  ti   = zeros(np,1);
  %% Loops
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
      for ic1=1:Nc
          param.c11 = vC(ic1);
          param.c12 = vC(ic1);
          for ic2=1:Nc
              param.c21 = vC(ic2);
              param.c22 = vC(ic2);
              for is = 1:Ns
                  %% Training model
                  svmStructs = OvR_TBSVM(Xtrain,tTrain,param,kern,vS(is));
                  %% classification
                  t_est = Predict_OvR_TBSVM(svmStructs,Xtest);
                  %% evalute erformance
                  [acm(ic1,ic2,is,f),fmem(ic1,ic2,is,f),~,...
                      gmem(ic1,ic2,is,f),~,gmem_o(ic1,ic2,is,f)] =...
                      performance_MC(tTest,t_est,svmStructs.labels);
              end
          end
      end
      ti(f) = toc;
  end
  %% Best Parameters
%   ac = squeeze(mean(acm,3));
%   gme = squeeze(mean(gmem,3));
  fme = squeeze(mean(fmem,4));
  gme_o = squeeze(mean(gmem_o,4));
  
  vfme = fme(:);
%   vgme = gme(:);
  vgme_o = gme_o(:);
%   vac = ac(:);

  Fmax = max(vfme);
  index = find(vfme==Fmax);
  [~,ind]=max(vgme_o(index));

  [ic1,ic2,is]=ind2sub([Nc,Nc,Ns],index(ind));

  param.c11 = vC(ic1);
  param.c12 = vC(ic1);
  param.c21 = vC(ic2);
  param.c22 = vC(ic2);
  BestPar.param = param;
  
  BestPar.sig = vS(is);
  kern.param = vS(is);
  
  BestPar.kern = kern;
  BestPar.performance = {squeeze(acm(ic1,ic2,is,:)),...
      squeeze(gmem(ic1,ic2,is,:)),squeeze(gmem_o(ic1,ic2,is,:)),...
      squeeze(fmem(ic1,ic2,is,:))}';