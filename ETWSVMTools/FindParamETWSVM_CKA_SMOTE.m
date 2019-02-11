function [BestPar,ti]=FindParamETWSVM_CKA_SMOTE(X,t,kern)
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
%       .k: best k-nn for SMOTE's algorithm
%       .performance: Accuracy, geometric mean and Fmeasure measurements 
%                     for parameters set selected.  

%%
np = 10; % np-fold 
index = A_crossval_bal(t,np); 


%%  Paramétros de entrenamiento
exp = [-9 -7 -5 -3 -1 1 3 5 7 9];
vk = [3 5 9 10 15];
vC = 2.^exp;
Nc = length(vC);
Nk = length(vk);
acm  = zeros(Nc,Nc,Nk,np);
gmem = zeros(Nc,Nc,Nk,np);
fmem = zeros(Nc,Nc,Nk,np);
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
  %% Metric Learning CKA
  [Xc1,~,label] = SeparateClassesBinary(Xtrain,tTrain);  
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
        [Xc11,Xc2,~] = SeparateClassesBinary(Xus,tus);
        L = double(bsxfun(@eq,l,l'));
        A = kMetricLearningMahalanobis([Xc11;Xc2],L,l,opts);
        kern.A = A;
  end
      %% we determinate the imbalanced rate
  nc1 = size(Xtrain(tTrain==label(1)),1);
  nc2 = size(Xtrain(tTrain==label(2)),1);
  Nm = floor(nc2/nc1-1)*100; 
  for ik=1:Nk
      Xsynthetic = SMOTE(Xc1,Nm,vk(ik));
      Xos = [Xtrain;Xsynthetic];
      tos = [tTrain;ones(size(Xsynthetic,1),1)];
      for ic1=1:Nc
          param.c11 = vC(ic1);
          param.c12 = vC(ic1);
          parfor ic2=1:Nc
                  %% Training model
                  svmStrucs = ETWSVMTraining(Xos,tos,param,kern,[],[],...
                      [],vC(ic2),vC(ic2));
                  %% classification
                  [t_est]=ETWSVMClassify(svmStrucs,Xtest);
                  %% evalute erformance
                  [acm(ic1,ic2,ik,f),gmem(ic1,ic2,ik,f),fmem(ic1,ic2,ik,f)] =...
                      Evaluate(tTest,t_est,svmStrucs.labels);
%               end
          end
      end
  end
  ti(f) = toc;
end
%% Best Parameters
% ac = squeeze(mean(acm,4));
gme = squeeze(mean(gmem,4));
fme = squeeze(mean(fmem,4));

vfme = fme(:);
vgme = gme(:);
% vac = ac(:);

Fmax = max(vfme);
index = find(vfme==Fmax);
[~,ind]=max(vgme(index));

[ic1,ic2,ik]=ind2sub([Nc Nc Nk],index(ind));

param.c11 = vC(ic1);
param.c12 = vC(ic1);
param.c21 = vC(ic2);
param.c22 = vC(ic2);
BestPar.param = param;
BestPar.k = vk(ik);
BestPar.performance = {squeeze(acm(ic1,ic2,ik,:)),squeeze(gmem(ic1,ic2,ik,:)),...
squeeze(fmem(ic1,ic2,ik,:))}';

% delete(gcp('nocreate'))