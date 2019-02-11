function struct = TBSVMTraining(X,t,kern,param,kernparam,c11,c12,c21,c22)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% Algorithm extracted of: Tian, Yingjie, et al. "Improved twin support
% vector machine." Science China Mathematics 57.2 (2014): 417-432. 
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
%                           .param: scalar array with kernel's parameters
%   c11, c12, c21, c22 (optional):  scalar regularization parameter for
%                                   parfor. Default all fixed in one value.
% OUTPUTS:
%   struct: exit struct with trained model.
%       .w1,.w2: orthogonal vector to minority hyperplane and majority one,
%               respectively. 
%       .b1,.b2: bias term of minority hyperplane and majority one,
%               respectively.
%       .alpha1,.alpha2: Lagrange multipliers of the minority hyperplane 
%                        and majority one, respectively.
%       .kern: kernel struct used to train.
%       .X : sorted samples matrix [X_+;X;-]
%       .nC1,.nC2: number of samples of minority class and majority one,
%                  respectively. 
%       .labels: cell or double array targets. Where the first target
%                correspond to minority class and second to majority one.
%       .norm_w1,.norm_w2: euclidean norm of the hyperplanes or surfaces (if is non-lineal)
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

if nargin>=5 && ~isempty(kernparam)
    kern.param = kernparam;
end

if nargin < 4
   param.c11 = 1; 
   param.c12 = 1;
   param.c21 = 1;
   param.c22 = 1;
    if nargin < 3
       kern.kernfunction = 'lin';
       if nargin < 2
            error('Insufficient arguments');
       end
    end
end

%% 
C11 = param.c11;
C12 = param.c12;
C21 = param.c21;
C22 = param.c22;

namekern = lower(kern.kernfunction);
%% Extracting the data matrixes
Class = unique(t);
if length(Class)~=2
    error('It is not a binary classification')
end
[X1,X2,labels] = SeparateClassesBinary(X,t);
nc1 = size(X1,1);
nc2 = size(X2,1);
D = [X1;X2];
%% Applied the kernel  function
if any(strcmp(namekern,{'lin','linear'}))
    S1 = [X1';ones(1,nc1)]; % P+1 x Nc1
    S2 = [X2';ones(1,nc2)]; % P+1 x Nc2
else
    K = ComputeKern(D,D,kern);
    S1 = [K(:,1:nc1);ones(1,nc1)];      % N+1 x Nc1
    S2 = [K(:,nc1+1:end);ones(1,nc2)];  % N+1 x Nc2
end

A1 = (C11*eye(size(S1,1))+S1*S1')\eye(size(S1,1));
A2 = (C12*eye(size(S2,1))+S2*S2')\eye(size(S2,1));
%% Hessian matrixes of the dual problems.
H1 = S2'*A1*S2;
H2 = S1'*A2*S1;
%% Slack values's regularization parameters
if isscalar(C21)
    C21 = C21*(nc1+nc2)/(2*nc2)*ones(nc2+1,1);
end
if isscalar(C22)
    C22 = C22*(nc1+nc2)/(2*nc1)*ones(nc1+1,1);
end
%% QPPS
options = optimoptions('quadprog','Display','off','Diagnostics','off');
alpha2 = quadprog(H1,-1*ones(nc2,1),[],[],[],[],...
        zeros(nc2,1),C21,[],options);
alpha1 = quadprog(H2,-1*ones(nc1,1),[],[],[],[],...
        zeros(nc1,1),C22,[],options);
%% Computed models's parameters
z1 = -A1*S2*alpha2; % z1=z_+=[w_+;b_+]
z2 = A2*S1*alpha1;  % z2=z_-=[w_-;b_-]
%% predict variable
w1 = z1(1:end-1);
w2 = z2(1:end-1);
if any(strcmp(namekern,{'lin','linear'}))
    struct.norm_w1 = sqrt(w1'*w1);
    struct.norm_w2 = sqrt(w2'*w2);
else
    struct.norm_w1 = sqrt(w1'*K*w1);
    struct.norm_w2 = sqrt(w2'*K*w2);
end

%% creating the struct
struct.w1 = w1;
struct.b1 = z1(end);
struct.w2 = w2;
struct.b2 = z2(end);
struct.alpha1 = alpha2;
struct.alpha2 = alpha1;
struct.kern = kern;
struct.ClassMin = labels(1);
struct.ClassMax = labels(2);
struct.X = D;
struct.nc1 = nc1;
struct.nc2 = nc2;

struct.labels = labels;