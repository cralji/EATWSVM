function [Xc1,Xc2,labels,NN]=SeparateClassesBinary(X,t)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USING:
%   [Xc1,Xc2,labels]=SeparacionClases(X,t)
% INPUTS:
%   X: Samples matrix R^{NxP}
%   t: targets vector R^{Nx1}
% OUTPUTS:
%   Xc1: Minoritory samples matrix, R^{N1xP}
%   Xc2: Majoritory samples matrix, R^{N2xP}
%   labels : cell or number array with the targets exists in t. First
%           element correspond the minority target and second correspond
%           the majority target.
%   NN: numeric array with number of samples for each class. 

    if nargin<2
        error('Requires at least two input arguments.')
    end

    labels=unique(t);
    nClass=length(labels);

    if (nClass>2) && (nClass<2)
        error('Disponible only for two classes')
    end
    %% identifies the minority class and majority one, and separates these 
    if iscell(t)
        B = [numel(find(string(t)==labels{1})) numel(find(string(t)==labels{2}))];
        if B(1)==B(2)
            Xc1=X(labels{1}==string(t),:);
            Xc2=X(labels{2}==string(t),:);
            labels = {labels{1};labels{2}};
        else
            [~,indmin] = min(B);
            [~,indmax] = max(B);
            Xc1=X(labels{indmin}==char(t),:);
            Xc2=X(labels{indmax}==char(t),:);
            labels = {labels{indmin};labels{indmax}};
        end
    else
        B = [numel(find(t==labels(1))) numel(find(t==labels(2)))];
        if B(1)==B(2)
            if any(labels==1)&&any(labels==-1)
                Xc1 = X(1 == t,:);
                Xc2 = X(-1== t,:);
                labels = [1;-1];
            else
                Xc1=X(labels(1)==t,:);
                Xc2=X(labels(2)==t,:);
            end
        else
            [~,indmin] = min(B);
            [~,indmax] = max(B);
            Xc1=X(labels(indmin)==t,:);
            Xc2=X(labels(indmax)==t,:);
            labels = [labels(indmin);labels(indmax)];
        end
    end
    NN = [size(Xc1,1) size(Xc2,1)];