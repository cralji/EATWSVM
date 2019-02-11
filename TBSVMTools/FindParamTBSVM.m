function BestPar = FindParamTBSVM(X,t,kern)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   BestPar=FindParamTBSVM_CKA(X,t,kern)
% INPUTS:
%   X: input samples matrix with which the parameters will be tuned.
%   t: targets array 
%   kern: struct kernel with fields:
%       .kernfunction: name kernel,  'lin': linearkernel
%                      and 'rbf': gaussian kernel. Default 'lin'
% OUTPUTS:
%   BestPar: exit struct with turned parameters. It has fields: 
%       .c11,.c12,.c21,.c22: The best regularization parameters set by
%                            F-measure and geometric mean.
%       .kern: the kernel struct with best parameter found.
%       .performance: Accuracy, geometric mean and Fmeasure measurements 
%                     for parameters set selected.  
np = 10;
index = A_crossval_bal(t,np);
warning('off','all')
%%  Paramétros de entrenamiento
exp = [-9 -7 -5 -3 -1 0 1 3 5 7 9];
vC = 2.^exp;
Nc = length(vC);

if ~any(strcmp(kern.kernfunction,{'lin','Lin','LIN'}))
  Nsig = 10;
  s0 = median(pdist(X));
  Vsig=linspace(0.1*s0,s0,Nsig)';
else
  Nsig = 1;
  Vsig = 1;
end

acm  = zeros(Nc,Nc,Nsig,np);
gmem = zeros(Nc,Nc,Nsig,np);
fmem = zeros(Nc,Nc,Nsig,np);
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
  %%
  Class = unique(tTrain);
  nSamples = [numel(find(tTrain==Class(1))) numel(find(tTrain==Class(2)))];
    [~,ind_min] = min(nSamples);
    ind_max = setdiff([1 2],ind_min);
    nc1 = nSamples(ind_min);
    nc2 = nSamples(ind_max);
  clear nSamples ind_min ind_max
  %% Loop
  for ic1=1:Nc
      param.c11 = vC(ic1);
      param.c12 = vC(ic1);
      for ic2=1:Nc
          param.c21 = vC(ic2)*ones(nc2+1,1);
          param.c22 = vC(ic2)*ones(nc1+1,1);
          parfor is=1:Nsig
              %% Training model
              svmStrucs = TBSVMTraining(Xtrain,tTrain,kern,param,Vsig(is));
              %% classification
              [t_est] = TBSVMClassify(Xtest,svmStrucs);
              %% evalute erformance
              [acm(ic1,ic2,is,f),gmem(ic1,ic2,is,f),fmem(ic1,ic2,is,f)] =...
                  Evaluate(tTest,t_est,svmStrucs.labels);
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

[ic1,ic2,is]=ind2sub([Nc,Nc,Nsig],index(ind));

param.c11 = vC(ic1);
param.c12 = vC(ic1);
param.c21 = vC(ic2);
param.c22 = vC(ic2);
BestPar.param = param;
kern.param = Vsig(is);
BestPar.kern = kern;

BestPar.performance = {squeeze(acm(ic1,ic2,is,:)),squeeze(gmem(ic1,ic2,is,:)),...
squeeze(fmem(ic1,ic2,is,:))}';