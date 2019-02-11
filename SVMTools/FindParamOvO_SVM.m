function [BestPar,ti] = FindParamOvO_SVM(X,t,kern)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   BestPar =FindParamOvO_SVM(X,t,kern)
% INPUTS:
%   X: input samples matrix with which the parameters will be tuned.
%   t: targets array 
%   kern: struct kernel with fields:
%       .function: name kernel,  'lin': linearkernel
%                      and 'rbf': gaussian kernel. Default 'lin'
% OUTPUTS:
%   BestPar: exit struct with turned parameters. It has fields: 
%       .C: The best regularization parameters set by
%                            F-measure and geometric mean.
%       .kern: kernel struct with best paremeter and regularization
%              parameters
%       .performance: performance measurements 
%                     for parameters set selected.  
  np = 10;
  index = A_crossval_bal(t,np);

  %%  Paramétros de entrenamiento
  exp = [-7 -5 -3 -1 0 1 3 5 7];
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
  acm  = zeros(Nc,Ns,np);
  gmem = zeros(Nc,Ns,np);
  fmem = zeros(Nc,Ns,np);
  gmem_o = zeros(Nc,Ns,np);
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
      for ic=1:Nc
          param.C = vC(ic);
          for is = 1:Ns
              %% Training model
              svmStructs = OvO_SVM(Xtrain,tTrain,param,kern,[],vS(is));
              %% classification
              t_est = Predict_OvO_SVM(svmStructs,Xtest);
              %% evalute erformance
              [acm(ic,is,f),fmem(ic,is,f),~,...
                  gmem(ic,is,f),~,gmem_o(ic,is,f)] =...
                  performance_MC(tTest,t_est,svmStructs.labels);
          end
      end
      ti(f) = toc;
  end
  %% Best Parameters
%   ac = squeeze(mean(acm,3));
%   gme = squeeze(mean(gmem,3));
  fme = squeeze(mean(fmem,3));
  gme_o = squeeze(mean(gmem_o,3));
  
  vfme = fme(:);
%   vgme = gme(:);
  vgme_o = gme_o(:);
%   vac = ac(:);

  Fmax = max(vfme);
  index = find(vfme==Fmax);
  [~,ind]=max(vgme_o(index));

  [ic,is]=ind2sub([Nc,Ns],index(ind));

  BestPar.C = vC(ic);  
  BestPar.sig = vS(is);
  kern.C = vC(ic);
  kern.param = vS(is);
  BestPar.kern = kern;
  BestPar.param.C = vC(ic);
  BestPar.performance = {squeeze(acm(ic,is,:)),...
      squeeze(gmem(ic,is,:)),squeeze(gmem_o(ic,is,:)),...
      squeeze(fmem(ic,is,:))}';