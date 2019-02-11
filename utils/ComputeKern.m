function K=ComputeKern(X1,X2,kern)
%@copyright Cristian Alfonso Jimenez Castano->e-mail:craljimenez@utp.edu.co
%Jimenez C., Alvarez A. and Gutierrez A. An enhanced twin support vector
%machine to support imbalanced data classification
% USAGE
%   K=ComputeKern(X1,X2,kern) 
% INPUTS: 
%   X1 \in R^{N x P}: first samples matrix [x_1' x_2' ... x_N']'
%   X2 \in R^{M x P}: second samples matrix [y_1' y_2' ... y_M']'
%   kern : Struct kernel with fields:
%       .kernfunction: name kernel to use. 'lin':linear kernel,'rbf':gaussian
%                      kernel.
%       .E: Transformated matrix of mahalanobis distance R^{PxQ} Q<=P. 
%           Default indenty matrix with Q=P. This is used for a Gaussian 
%           kernel with covariance matrix no-isotropic (C^-1=E*E').
%       .param: number array with parameters kernel [p1 p2 p3 ... pn].
% OUTPUTS:
%         K \in R^{N x M}: matrix kernel with elements K_{ij}=k(x_i,y_j)

    
    namefunction=kern.kernfunction;
    names = fieldnames(kern);
    if ~any(strcmp(namefunction,{'lin','LIN','Lin'}))
        if any(strcmp(names,'E'))
            E = kern.E;
        else
            E = eye(size(X1,2));
        end
        if any(strcmp(names,'param'))
            param=kern.param;
        else
            param = 1;
            %error('No existe opci�n para parametro kernel (.param o .A)')
        end
    end
    
    switch namefunction
        case {'rbf','RBF'}
            Y1 = X1*E;
            Y2 = X2*E;                
            if length(param)==1
                K=exp(-pdist2(Y1,Y2).^2/(2*param^2));
            else
                K=param(2)*exp(-pdist2(Y1,Y2).^2/(2*param(1)^2));
            end
        case {'lin','LIN','Lin'}
            K=X1*X2';
        otherwise
            error('Tipo de kernel no disponible')
    end