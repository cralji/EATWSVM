function ETWSVM_Strucs = ETWSVMTraining(X,t,param,kern,paramkern,c11,c12,c21,c22)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%   ETWSVM_Strucs = ETWSVMTraining(X,t,param,kern,sig,c22,c11)
% INPUTS:
%   X: Samples matrix R^{NxP} 
%   t: target vector R^{Nx1} 
%   param: struct with scalar fields:
%          .c11, .c12, .c21 and .c22: These are regularization parameters.
%          Default all in one value.
%          .TypeClassifier: if you want is binary training this field must
%                          fix in 'binary', and if you want is multi-class 
%                          training this fielt must fix how 'multiclass'.
%                          Default 'binary'
%                          
%   kern: struct with fields
%                           .kernfunction: name kernel,  'lin': linearkernel
%                           and 'rbf': gaussian kernel. Default 'lin'
%                           .E: Transformated matrix of mahalanobis distance
%                               R^{PxQ} Q<=P. Default indenty matrix with
%                               Q=P.
%                           .param: scalar array with kernel's parameters
%   c11, c12, c21, c22 (optional):  scalar regularization parameter for
%                                   parfor. Default all fixed in one value.
% OUTPUTS:
%   ETWSVM_Strucs: exit structure. if param.TypeClassifier is equal to 
%                  'binary', then this struct will have fields:
%                  .svmPlus: struct with the mimonity hyperplane. This have
%                  fields:
%                           .S: Kernel matrix extended between the minority
%                               samples K_{11}+1_1*1_1'.
%                           .Scross: Kernel matrix extended  between
%                                    minority samples and majority samples,
%                                    K_{12}+1_1*1_2'.
%                           .X: minority samples matrix (R^{N1xP}).
%                           .B: Inverse Gram matrix of minority samples.
%                           .alpha: vector of the Lagrange multipliers of 
%                                   minority hyperplane (R^{N2x1})
%                           .ww: euclidean norm to square of the minority 
%                                hyperplane's weighted vector 
%                  .svmMinus: struct with the majority hyperplane. This have
%                  fields:
%                           .S: Kernel matrix extended between the majority
%                               samples K_{22}+1_2*1_2'.
%                           .Scross: Kernel matrix extended  between
%                                    minority samples and majority samples,
%                                    K_{21}+1_2*1_1'.
%                           .X: majority samples matrix (R^{N2xP}).
%                           .B: Inverse Gram matrix of majority samples.
%                           .alpha: vector of the Lagrange multipliers of 
%                                   majority hyperplane (R^{N1x1})
%                           .ww: euclidean norm to square of the majority
%                                hyperplane's weighted vector.
%                   .param: struct with regularization parameters
%                   .kern: kernel struct used to train.
%                   .labels: cell or double array targets. Where the
%                            first target correspond to minority class and 
%                            second to majority one.
%               if param.TypeClassifier is equal to 'multiclass', then this
%               struct will have fields:
%                       .S: Kernel matrix extended between the minority
%                           samples K_{11}+1_1*1_1'.
%                       .Scross: Kernel matrix extended  between
%                                minority samples and majority samples,
%                                K_{12}+1_1*1_2'.
%                        .X: minority samples matrix (R^{N1xP}).
%                        .X2: majority samples matrix (R^{N2xP}).
%                        .B: Inverse Gram matrix of minority samples.
%                        .alpha: vector of the Lagrange multipliers of 
%                                minority hyperplane (R^{N2x1})
%                        .ww: euclidean norm to square of the minority 
%                             hyperplane's weighted vector
%               
    
%%
if nargin>=6 && ~isempty(c11)
    param.c11 = c11;
end
if nargin>=7 && ~isempty(c12)
    param.c12 = c12;
end
if nargin>=8 && ~isempty(c21)
    param.c21 = c21;
end
if nargin==9 && ~isempty(c22)
    param.c22 = c22;
end

if nargin>=5 && ~isempty(paramkern)
    kern.param = paramkern;
end
if nargin<4
    kern.kernfunction='lin';
    if nargin<3
        param.c11 = 1;
        param.c12 = 1;
        param.c21 = 1;
        param.c22 = 1;
        if nargin<2
            error('Argumentos insuficientes.')
        end
    end
end

%% initial variables
c11 = param.c11;
c21 = param.c21;

if any(strcmp(lower(fieldnames(param)),'typeclassifier'))
    TypeClassifier = lower(param.typeclassifier);        
else
    TypeClassifier = 'binary';
end    
if strcmp(TypeClassifier,'binary')
    c12 = param.c12;
    c22 = param.c22;
end
N = size(X,1);
if ~iscell(t)
    if any(t==1)&&any(t==-1)
        labels = [1;-1];
        Xc1 = X(t==labels(1),:); Xc2 = X(t==labels(2),:);
    else
        [Xc1,Xc2,labels] = SeparateClassesBinary(X,t);
    end
else
    if any(string(t)==string(1)) && any(string(t)==string(-1))
        labels = {'1';'-1'};
        Xc1 = X(string(t)==string(1));
        Xc2 = X(string(t)==string(-1));
    else
        [Xc1,Xc2,labels] = SeparateClassesBinary(X,t);
    end
end
nc1=size(Xc1,1);
nc2=size(Xc2,1);
D = [Xc1;Xc2];
%% compute kernel matrix
K = ComputeKern(D,D,kern);
K11 = K(1:nc1,1:nc1);
K12 = K(1:nc1,nc1+1:end);
K21 = K(nc1+1:end,1:nc1);
K22 = K(nc1+1:end,nc1+1:end);

S12 = K12 + ones(nc1,nc2);
S21 = S12';
S11 = K11 + ones(nc1,nc1);
S22 = K22 + ones(nc2,nc2);

B1 = ((c11)*eye(nc1,nc1)+S11)\eye(nc1,nc1); % Inverse Gramm matrix
H1 = (1/c11)*(S22 - S21*B1*S12); % Hessian Matrix of the minority hyperplane
H1 = 0.5*(H1' + H1);  % ensure symmetry

%%
if any(any(isnan(H1)))
    error('H1 have components NaN');
end

%% optimization options
options = optimoptions('quadprog','Display','off','Diagnostics','off');
if isscalar(c21)
    C21 = c21*ones(nc2,1)*(N/(2*nc2));
elseif ismatrix(c21)
    C21 = c21;
end

%% QPP Minority hyperplane w_+ b_+, 
alpha=quadprog(H1,-1*ones(nc2,1),[],[],[],[],...
    zeros(nc2,1),C21,[],options);

%% ||w_+||^2
ww1 = (1/(c11*c11))*alpha'*(K22 - 2*K21*B1*S12 + S21*B1*K11*B1*S12)*alpha;
ww1(ww1<=0) = 1e-8;

svmPlus.S = S11;
svmPlus.Scross = S12;
svmPlus.X = Xc1;
svmPlus.B = B1;
svmPlus.alpha = alpha;
svmPlus.ww = ww1;

%% for majority hyperplane
if ~strcmp(TypeClassifier,'multiclass')
    B2 = ((c12)*eye(nc2,nc2)+S22)\eye(nc2,nc2);
    H2 = (1/c12)*(S11 - S12*B2*S21);
    H2 = 0.5*(H2' + H2);
    if any(any(isnan(H2)))
        error('H2 tiene componentes NaN');
    end

    if isscalar(c22)
        C22 = c22*ones(nc1,1)*(N/(2*nc1));
    elseif (c22)
        C22 = c22;
    end
    %% QPP Majority hyperplane w_- b_-
    gamma=quadprog(H2,-1*ones(nc1,1),[],[],[],[],...
        zeros(nc1,1),C22,[],options);
    ww2 = (1/(c12*c12))*gamma'*(K11 - 2*K12*B2*S21 + S12*B2*K22*B2*S21)*gamma;
    ww2(ww2<=0) = 1e-8;

    svmMinus.S = S22;
    svmMinus.Scross = S21;
    svmMinus.X = Xc2;
    svmMinus.B = B2;
    svmMinus.alpha = gamma;
    svmMinus.ww = ww2;

    ETWSVM_Strucs.Minus= svmMinus;
    ETWSVM_Strucs.param = param;
    ETWSVM_Strucs.kern = kern;
    ETWSVM_Strucs.labels = labels;

    ETWSVM_Strucs.Plus = svmPlus;
else
    svmPlus.X2 = Xc2;
    ETWSVM_Strucs = svmPlus;
end