function [BestPar,ti]=FindParamSVM_SMOTE(X,t,kern)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   BestPar=FindParamSVM_SMOTE(X,t,kern)
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
%       .k: best k-nn for SMOTE's algorithm

%%
np=10;
index=A_crossval_bal(t,np);

exp1 = [-9 -7 -5 -3 -1 0 1 3 5 7 9]';
c = 2.^exp1;
% c=linspace(1e-6,1e9,20)';
k=[3 5 7 9 11 13 15]';

% acm=zeros(Nc,Nsig,Nk,np,3);
%% Paramétros de entrenamiento
if any(strcmp(kern.function,{'rbf','RBF'}))
    Nsig = 10;
    s0=median(pdist(X));
    sig=linspace(0.1*s0,s0,Nsig)';
else
    sig=1;
    Nsig=1;
end
Nk=size(k,1);
Nc=size(c,1);
ac = zeros(Nc,Nsig,Nk,np);
gm = zeros(Nc,Nsig,Nk,np);
fm = zeros(Nc,Nsig,Nk,np);
ti = zeros(np,1);
SIG = zeros(Nsig,np);
%% Loops

for f=1:np
    %% Partition
    tic
    fprintf('\t %i \n',f)
    indTrain=find(index~=f);
    indTest=find(index==f);
    
    Xtrain=X(indTrain,:);
    Xtest=X(indTest,:);
   
    tTrain=t(indTrain);
    tTest=t(indTest);
    
    
    if numel(unique(tTrain))~=1 
    
    
    %% Separete of class
    [Xmin,Xmax,labelsminmaj]=SeparateClassesBinary(Xtrain,tTrain);
    nmin=size(Xmin,1);
    nmax=size(Xmax,1);
    N=((nmax-nmin)/nmin)*100;
    
    
    %% Loop main
        for ik=1:length(k)
            Synthetic = SMOTE(Xmin,N,k(ik));
            Xos = [Xtrain;Synthetic];
            tos=[tTrain;repmat(labelsminmaj(1),size(Synthetic,1),1)];
            %%
            for ic=1:size(c,1)
                kern.C=c(ic);
                parfor l=1:size(sig,1)
                    t_est=SVMTraining(Xos,tos,Xtest,kern,sig(l));
                    [ac(ic,l,ik,f),gm(ic,l,ik,f),fm(ic,l,ik,f)] = Evaluate(...
                        tTest,t_est,labelsminmaj); % performance model
                end
            end
        end
    else
        fprintf('\t nClass training is 1 \n');
    end
    ti(f) = toc;
    clear Xtrain Xtest tTrain tTest indTest indTrain 
end
%% Best Parame
acc = squeeze(mean(ac,4));
gme = squeeze(mean(gm,4));
fme = squeeze(mean(fm,4));

vfme = fme(:);
vgme = gme(:);
vac = acc(:);

Fmax = max(vfme);
index = find(vfme==Fmax);
[~,ind] = max(vgme(index));

[ic,l,ik] = ind2sub([Nc,Nsig,Nk],index(ind));

BestPar.C = c(ic);
BestPar.param = sig(l);
kern.C = c(ic); kern.param = sig(l);
BestPar.kern = kern;
BestPar.k = k(ik);
BestPar.performance = {squeeze(ac(ic,l,ik,:)),squeeze(gm(ic,l,ik,:)),...
squeeze(fm(ic,l,ik,:))}';
    