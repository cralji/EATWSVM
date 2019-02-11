function [BestPar] = FindParamSVM(X,t,kern)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   BestPar=FindParamSVM(X,t,kern)
% INPUTS:
%   X: input samples matrix with which the parameters will be tuned.
%   t: targets array 
%   kern: kernel struct with fields:
%       .C: regularization parameters
%       .param: kernel function's parameters
%       .function: kernel function's name. 'linear'(Default) and 'rbf'
% OUTPUTS:
%   BestPar: exit struct with turned parameters. It has fields: 
%       .C: The best regularization parameter by F-measure and geometric 
%           mean.
%       .kern: the kernel struct with best parameter found.
%       .performance: Accuracy, geometric mean and Fmeasure measurements 
%                     for parameters set selected.  
np=10;
index=A_crossval_bal(t,np);
%%
exp1 = [-9 -7 -5 -3 -1 0 1 3 5 7 9]';
c = 2.^exp1;

if any(strcmp(lower(kern.function),'rbf'))
    Nsig = 10;
    s0=median(pdist(X));
    sig=linspace(0.1*s0,s0,Nsig)';
else
    sig=1;
    Nsig=1;
end
Nc=size(c,1);
acm = zeros(Nc,Nsig,np);
gm = zeros(Nc,Nsig,np);
fm = zeros(Nc,Nsig,np);
ti = zeros(np,1);
%% Loops
[~,~,labels]=SeparateClassesBinary(X,t);
for f=1:np
    %% Partition
    tic
    fprintf('\t %i \n',f)
    indTrain=index~=f;
    indTest=index==f;
    
    Xtrain=X(indTrain,:);
    Xtest=X(indTest,:);
    
    tTrain=t(indTrain);
    tTest=t(indTest);    
    %% Loop
    for ic=1:size(c,1)
        kern.C=c(ic);
        parfor l = 1:Nsig
            t_est = SVMTraining(Xtrain,tTrain,Xtest,kern,sig(l));
            [acm(ic,l,f),gm(ic,l,f),fm(ic,l,f)] = Evaluate(...
                tTest,t_est,labels); % performance model
        end
    end
    ti(f) = toc;
end
%% Best Parameters
%   ac = squeeze(mean(acm,3));
  gme = squeeze(mean(gm,3));
  fme = squeeze(mean(fm,3));
  
  vfme = fme(:);
  vgme = gme(:);
%   vac = ac(:);

  Fmax = max(vfme);
  index = find(vfme==Fmax);
  [~,ind]=max(vgme(index));
  
  [ic,l]=ind2sub([Nc,Nsig],index(ind));

  BestPar.C=c(ic);
  kern.C = c(ic); kern.param = sig(l);
  BestPar.kern = kern;
  BestPar.performance = {squeeze(acm(ic,l,:)),squeeze(gm(ic,l,:)),...
    squeeze(fm(ic,l,:))}';