function BestPar=FindParamETWSVM_CKA(X,t,kern)
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
%       .performance: Accuracy, geometric mean and Fmeasure measurements 
%                     for parameters set selected.

  np=10;
  index=A_crossval_bal(t,np);
  %%  Paramétros de entrenamiento
  exp = [-9 -7 -5 -3 -1 0 1 3 5 7 9];
  vC = 2.^exp;
  Nc = length(vC);
  
  acm  = zeros(Nc,Nc,np);
  gmem = zeros(Nc,Nc,np);
  fmem = zeros(Nc,Nc,np);
  ti   = zeros(np,1);
  [~,~,label] = SeparateClassesBinary(X,t);
  %% Loops
  for f=1:np
      %% Partition
      tic
      fprintf('\t %d \n',f)
      Xtrain = X(index~=f,:);
      Xtest = X(index==f,:);

      tTrain = t(index~=f);
      tTest = t(index==f);
      %% Metric Learning CKA
      if any(strcmp(kern.kernfunction,{'RBF','rbf'}))
            opts = ckaoptimset();
            opts.showCommandLine=false;
            opts.showWindow=false;
            opts.maxiter = 120;
            opts.Q = 0.95;
            [Xus,tus] = US_vnn(Xtrain,tTrain,8);
            nc1 = size(Xus(tus==label(1)),1);
            nc2 = size(Xus(tus==label(2)),1);
            l = [ones(nc1,1);-1*ones(nc2,1)];
            [Xc1,Xc2,~] = SeparateClassesBinary(Xus,tus);
            L = double(bsxfun(@eq,l,l'));
            A = kMetricLearningMahalanobis([Xc1;Xc2],L,l,opts);
            kern.A = A;
      else
          A = eye(size(X,2));
      end
      %% Loop
%       param.c2 = 1;
      for ic1=1:Nc
          param.c11 = vC(ic1);
          param.c12 = vC(ic1);
          for ic2=1:Nc
              %% Training model
              svmStrucs = ETWSVMTraining(Xtrain,tTrain,param,kern,[],[],...
                  [],vC(ic2),vC(ic2));
              %% classification
              [t_est]=ETWSVMClassify(svmStrucs,Xtest);
              %% evalute performance
              [acm(ic1,ic2,f),gmem(ic1,ic2,f),fmem(ic1,ic2,f)] =...
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

  [ic1,ic2] = ind2sub([Nc Nc],index(ind));

  param.c11 = vC(ic1);
  param.c12 = vC(ic1);
  param.c21 = vC(ic2);
  param.c22 = vC(ic2);
  BestPar.param = param;
  BestPar.performance = {squeeze(acm(ic1,ic2,:)),squeeze(gmem(ic1,ic2,:)),...
    squeeze(fmem(ic1,ic2,:))}';
% delete(gcp('nocreate'))