function [t_est,F,D] = TBSVMClassify(Xtest,struct)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% Algorithm extracted of: Tian, Yingjie, et al. "Improved twin support
% vector machine." Science China Mathematics 57.2 (2014): 417-432. 
% USAGE:
%   [t_est,F,D] = TBSVMClassify(Xtest,struct)
% INPUTS:
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
%       .norm_w1,.norm_w2: euclidean norm of the hyperplanes or surfaces
%                          (if is non-lineal)
%   Xtest: test sample matrix, sample for row.
% OUTPUTS:
%   t_est: estimate targets array
%   F : scores matrix. The first column are the score the estimate targets
%       for minority hyperplane and second column corresponds to majority
%       hyperplane.
%   D : distances matrix. the first column correspond the norm distance of
%       the test samples to minority hyperplane and second column correspond
%       to majority hyperplane.
%% extracting neccesary variables
kern = struct.kern;
D = struct.X;
w1 = struct.w1;
w2 = struct.w2;
b1 = struct.b1;
b2 = struct.b2;
namekern = lower(kern.kernfunction);
%%
nt = size(Xtest,1);
if any(strcmp(namekern,{'lin','linear'}))
    norm_w1 = struct.norm_w1;
    norm_w2 = struct.norm_w2;
    F1 = (Xtest*w1 + b1*ones(nt,1))./norm_w1;
    F2 = (Xtest*w2 + b2*ones(nt,1))./norm_w2;
else
    den1 = struct.norm_w1;
    den2 = struct.norm_w2;
    
    K = ComputeKern(Xtest,D,kern);
    F1 = abs(K*w1 + b1*ones(nt,1));
    F2 = abs(K*w2 + b2*ones(nt,1));
    D1 = F1./den1;
    D2 = F2./den2;
end
F = [F1 F2];
D = [D1 D2];

t_est = sign(D2-D1);
t_est(t_est==-1) = struct.ClassMax;
t_est(t_est==1) = struct.ClassMin;
% t_est(t_est==0) = 1; % evitar problemas
