function [t_est,F,D]=ETWSVMClassify(ETWSVMstrucs,Xtest)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE:
%   [t_est,F,D]=ETWSVMClassify(svmStrucs,Xtest)
% INPUTS:
%   ETWSVMstrucs: struct of the training with fields:
%       .svmPlus: struct with the mimonity hyperplane. This have
%                  fields:
%                .S: Kernel matrix extended between the minority
%                    samples K_{11}+1_1*1_1'.
%                .Scross: Kernel matrix extended  between
%                         minority samples and majority samples,
%                         K_{12}+1_1*1_2'.
%                .X: minority samples matrix (R^{N1xP}).
%                .B: Inverse Gram matrix of minority samples.
%                .alpha: vector of the Lagrange multipliers of 
%                        minority hyperplane (R^{N2x1})
%                .ww: euclidean norm to square of the minority 
%                     hyperplane's weighted vector ||w_1||^2
%       .svmMinus: struct with the majority hyperplane. This have
%                  fields:
%                .S: Kernel matrix extended between the majority
%                    samples K_{22}+1_2*1_2'.
%                .Scross: Kernel matrix extended  between
%                         minority samples and majority samples,
%                         K_{21}+1_2*1_1'.
%                .X: majority samples matrix (R^{N2xP}).
%                .B: Inverse Gram matrix of majority samples.
%                .alpha: vector of the Lagrange multipliers of 
%                        majority hyperplane (R^{N1x1})
%                .ww: euclidean norm to square of the majority
%                     hyperplane's weighted vector ||w_2||^2.
%       .param: struct with regularization parameters
%       .kern: kernel struct used to train.
%       .labels: cell or double array targets. Where the
%                first target correspond to minority class and 
%                second to majority one.
%   Xtest: test sample matrix, sample for row.
% OUTPUTS:
%   t_est: estimate targets array
%   F : scores matrix. The first column are the score the estimate targets
%       for minority hyperplane and second column corresponds to majority
%       hyperplane.
%   D : distances matrix. the first column correspond the norm distance of
%       the test samples to minority hyperplane and second column correspond
%       to majority hyperplane.

    %% Extract necessary variables
    svmPlus = ETWSVMstrucs.Plus;
    svmMinus= ETWSVMstrucs.Minus;
    param = ETWSVMstrucs.param;
    kern = ETWSVMstrucs.kern;
    labels = ETWSVMstrucs.labels;

    c11 = param.c11;
    c12 = param.c12;
    B1 = svmPlus.B;
    B2 = svmMinus.B;

    alpha = svmPlus.alpha;
    gamma = svmMinus.alpha;

%     S11 = svmPlus.S;
    S12 = svmPlus.Scross;
%     S22 = svmMinus.S;
    S21 = svmMinus.Scross;

    Xc1 = svmPlus.X; nc1 = size(Xc1,1);
    Xc2 = svmMinus.X; nc2 = size(Xc2,1);

    ww1 = sqrt(svmPlus.ww);
    ww2 = sqrt(svmMinus.ww);
    
    %% It compute score
    nt = size(Xtest,1);

    Knew_1 = ComputeKern(Xtest,Xc1,kern) + ones(nt,nc1);
    Knew_2 = ComputeKern(Xtest,Xc2,kern) + ones(nt,nc2);
    f1 = (1/c11)*(Knew_2 - Knew_1*B1*S12)*alpha*(-1);
    f2 = (1/c12)*(Knew_1 - Knew_2*B2*S21)*gamma*(1);
    F = [f1 f2];
    D = [abs(f1)./ww1 abs(f2)./ww2]; % norm distance to minority hyperplane and majority one.

    t_est = sign( D(:,2) - D(:,1) );
    t_est(t_est==0) = 1;
    %% we recover the original targets
    if ~iscell(labels)
        t_est(t_est==1) = labels(1);
        t_est(t_est==-1)= labels(2);
    else
        t = cell(nt,1);
        for i=1:nt
            if t_est(i)==1
                t{i} = labels(1);
            else
                t{i} = labels(2);
            end
        end
        clear t_est
        t_est = t;
    end