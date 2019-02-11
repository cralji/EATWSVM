function [BestPar,ti]=FindParamWLTSVM(X,t,tipokern)
% @craljimenez: Cristian Jimenez. Universidad Tecnologica de Pereira
% e-mail: craljimenez@utp.edu.co
nc=2; % work cluster
np=10;
index=A_crossval_bal(t,np);
%% Parameters


%%
%% Paramétros de entrenamiento
if any(strcmp(tipokern,{'rbf','RBF'}))
    Nsig = 10;
    s0=median(pdist(X));
    sig=linspace(0.1*s0,s0,Nsig)';
else
    sig=1;
    Nsig=1;
end
        

exp1 = [-5 -3 -2 0 2 5 7 9]';
c = 2.^exp1;
Nc=length(c);

ac = zeros(Nc,Nc,Nsig,np);
gm = zeros(Nc,Nc,Nsig,np);
fm = zeros(Nc,Nc,Nsig,np);
ti = zeros(np,1);
SIG = zeros(Nsig,np);

FunPara.kerfPara.type = tipokern;
%% Loops
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
    
    [Xtrain,mu,st]=zscore(Xtrain);
    st(abs(st)<1e-13)=1; % normalization's problem
    Xtest=(Xtest-repmat(mu,size(Xtest,1),1))./repmat(st,size(Xtest,1),1);
    
    %% Class separete
    [Xc1,Xc2,labels]=SeparateClassesBinary(Xtrain,tTrain); % Separa los datos de entrada la clase Minoritaria,Xc1, de la clase mayoritaria, Xc2
    DataTrain.A=Xc1;
    DataTrain.B=Xc2;
    % las etiquetas (t) de Xc1 se asumen +1, y de Xc2 -1.
    for ic1=1:length(c)
    for ic2=1:length(c)
        
        
        FunPara.p1=c(ic1);
        FunPara.p2=c(ic2);
        %% Loop
%         parfor (l=1:size(sig,1),nc)
        for l=1:size(sig,1)
%             FunPara.kerfPara.pars=sig(l);
            [t_est]=WLTSVM(Xtest,DataTrain,FunPara,sig(l)); % out= Accu Gmean Fmeasure
            if ~any(isnan(t_est))
            [ac(ic1,ic2,l,f),gm(ic1,ic2,l,f),fm(ic1,ic2,l,f)] = Evaluate(...
                        tTest,t_est,labels); % performance model
            else
                ac(ic1,ic2,l,f) = 0;
                gm(ic1,ic2,l,f) = 0;
                fm(ic1,ic2,l,f) = 0;
            end
        end
    end
    end
    ti(f) = toc;
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
[~,ind]=max(vgme(index));

[ic1,ic2,l]=ind2sub([Nc,Nc,Nsig],index(ind));

BestPar.c1 = c(ic1);
BestPar.c2 = c(ic2);
BestPar.param = sig(l);
BestPar.performance = {squeeze(ac(ic1,ic2,l,:)),squeeze(gm(ic1,ic2,l,:)),...
squeeze(fm(ic1,ic2,l,:))}';
    