function [BestPar,ti]=FindParamETWSVM_CKA_SMOTE_fixedC(X,t,kern)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   BestPar=FindParamETWSVM_CKA_SMOTE_fixedC(X,t,kern)
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
%       .k: best k-nn for SMOTE's algorithm
%       .performance: Accuracy, geometric mean and Fmeasure measurements 
%                     for parameters set selected.  
  np=10;
  index=A_crossval_bal(t,np);
  
  if any(strcmp(kern.kernfunction,{'RBF','rbf'}))
      Ns = 10;
      s0 = median(pdist(X));
      Vs =linspace(0.1*s0,s0,Ns);
  else
      Ns = 1;
      Vs = 1;
  end
  vK = [1 3 5 10 15];
  Nk = length(vK);
  %%  Paramétros de entrenamiento
  ti = zeros(np,1);
  acm  = zeros(Nk,Ns,np);
  gmem = zeros(Nk,Ns,np);
  fmem = zeros(Nk,Ns,np);
  
  param.c11 = 0.001; 
  param.c12 = 0.001;
  param.c21 = 1;
  param.c22 = 1;
  
  %% Loops
  for f=1:np
      %% Partition
      fprintf('\t %d \n',f)
      Xtrain = X(index~=f,:);
      Xtest = X(index==f,:);

      tTrain = t(index~=f);
      tTest = t(index==f);
      tic
      %% loop
      for ik = 1:Nk
          [X1,~,label,NN] = SeparateClassesBinary(Xtrain,tTrain);
          m = (NN(2)/NN(1) - 1)*100;
          if m<=1
              m = 100;
          end
          Xsyn = SMOTE(X1,m,vK(ik));
          Xos = [Xtrain;Xsyn];
          tos = [tTrain;label(1)*ones(size(Xsyn,1),1)];
          for is = 1:Ns
              %% Training model
              svmStrucs = ETWSVMTraining(Xos,tos,param,kern,Vs(is));
              %% classification
              [t_est]=ETWSVMClassify(svmStrucs,Xtest);
              %% evalute performance
              [acm(ik,is,f),gmem(ik,is,f),fmem(ik,is,f)] =...
                  Evaluate(tTest,t_est,svmStrucs.labels);
          end
      end
      ti(f) = toc;
  end
  %% Best Parameters
%   ac = squeeze(mean(acm,3));
  gme = squeeze(mean(gmem,3));
  fme = squeeze(mean(fmem,3));

  vfme = fme(:);
  vgme = gme(:);
%   vac = ac(:);

  Fmax = max(vfme);
  index = find(vfme==Fmax);
  [~,ind]=max(vgme(index));

  [ik,is]=ind2sub([Nk,Ns],index(ind));

  BestPar.param = param;
  kern.param = Vs(is);
  BestPar.kern = kern;
  BestPar.k = vK(ik);
  
  BestPar.performance = {squeeze(acm(ik,is,:)),squeeze(gmem(ik,is,:)),...
    squeeze(fmem(ik,is,:))}';