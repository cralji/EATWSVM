#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:23:09 2020

@author: craljimenez@utp.edu.co, Cristian Alfonso Jimenez Castaño

    C. Jimenez-Castaño, A. Alvarez-Meza, A. Orozco-Gutierrez,
    Enhanced automatic twin support vector machine for imbalanced data classification,
    Pattern Recognition,
    Volume 107,
    2020,
    107442,
    ISSN 0031-3203,
    https://doi.org/10.1016/j.patcog.2020.107442.
    (http://www.sciencedirect.com/science/article/pii/S0031320320302454)
    Abstract: Most of the classification approaches assume that the sample distribution among classes is balanced. Still, such an assumption leads to biased performance over the majority class. This paper proposes an enhanced automatic twin support vector machine – (EATWSVM) to deal with imbalanced data, which incorporates a kernel representation within a TWSVM-based optimization. To learn the kernel function, we impose a Gaussian similarity, ruled by a Mahalanobis distance, and couple a centered kernel alignment-based approach to improving the data separability. Besides, we suggest a suitable range to fix the regularization parameters concerning both the dataset’ imbalance ratio and overlap. Lastly, we adopt One-vs-One and One-vs-Rest frameworks to extend our EATWSVM formulation for multi-class tasks. Obtained results on synthetic and real-world datasets show that our approach outperforms state-of-the-art methods concerning classification performance and training time.
    Keywords: Imbalanced data; Kernel methods; Twin support vector machines

reuirements:
    cvxopt package
    sklearn
    qpsolvers package
    scipy
    numpy

""" 

#%% ********************************* import librariers *********************************


import scipy.io as io
import numpy as np
# import Classes
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.base import BaseEstimator,ClassifierMixin
from qpsolvers import solve_qp
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn import metrics

from CKA_Adam_class import CKA_Adam as CKA
#%% functions and classes 
def Targets2Labels(t):
    """
    util funtion, change the target names to labels (int)

    Parameters
    ----------
    t : numpy array, with targets 
        DESCRIPTION.

    Returns
    -------
    y : numerical vector with appropriate labels
        DESCRIPTION.
    labels : target's names, the first is correspond to label +1
        DESCRIPTION.

    """
    # t = self.t
    if type(t) == list:
        t = np.array(t)
    targets = np.unique(t)
    nC = len(targets)
    Nc = []
    N = t.shape[0]
    y = np.zeros((N,1))
       
    for i in range(nC):
        Nc.append(np.sum(t==targets[i]))
    
    if nC == 2:
        if Nc[0] != Nc[1]:
            indMajority = np.argmax(Nc)
            indMinority = np.argmin(Nc)
            y[t == targets[indMinority]] = 1
            y[t == targets[indMajority]] = -1
            labels = np.array([targets[indMinority],targets[indMajority]])
        elif Nc[0] == Nc[1]:
            y[t == targets[0]] = 1
            y[t == targets[1]] = -1
            labels = targets
    elif nC>2:
        for i in range(nC):
            y[t == targets[i]] = i+1
            labels = targets
    # self.labels = labels
    # self.y = y
    
    return y,labels

class kernel:
    def __init__(self,kernfunction = 'lin', kernparam = 1,E = np.eye(2),are_samples = True):
        self.kernfunction = kernfunction
        self.kernparam = kernparam
        self.kernparamGlobal = kernparam
        self.kernfunctionGlobal = kernfunction
        self.are_samples = are_samples
        self.E = E
        
    def ComputeKern(self,Data):
        """
        Parameters
        ----------
        Data : tuple
            if tuple is (X1,X2), where X1 in R^{n1 x P} and X2 in R^{n2 x P}, and P 
            is of number of features; compute kernel matrix in R^{n1 x n2}.if tuple is 
            (Dist), then compute kernel matrix from distance matrice Dist in R^{n1 x n2}
        Returns
        -------
        K : narray. 

        """
        
        kernfunction = self.kernfunction
        kernparam = self.kernparam
        E = self.E
        
        if len(Data) == 2:
            X1 = Data[0]
            X2 = Data[1]
        else:
            Dist = Data[0]
        if self.are_samples:
            if np.all(E == np.eye(2)):
                E = np.eye(X1.shape[1])
            
            Y1 = np.dot(X1,E);
            Y2 = np.dot(X2,E);
            
            if kernfunction.lower() == 'lin' or kernfunction.lower() == 'linear':
                K = np.dot(Y1,Y2.T)
            elif kernfunction.lower() == 'rbf':
                K = np.exp(-.5*(cdist(Y1,Y2)**2)/((kernparam**2)))
            elif kernfunction.lower() == 'nrbf' or kernfunction.lower() == 'gaussian':
                K = (1/(np.sqrt(2*np.pi)*kernparam))*np.exp(-.5*(cdist(Y1,Y2)**2)/(2*(kernparam**2)))
            else:
                K = np.zeros((Y1.shape[1],Y1.shape[2]))
        else:
            if kernfunction == None:
                kernfunction = self.kernfunctionGlobal
            if kernparam == None:
                kernparam = self.kernparamGlobal
            
            if kernfunction.lower() == 'lin' or kernfunction.lower() == 'linear':
                K = Dist
            elif kernfunction.lower() == 'rbf':
                K = np.exp(-.5*(Dist**2)/(2*(kernparam**2)))
            elif kernfunction.lower() == 'nrbf' or kernfunction.lower() == 'gaussian':
                K = (1/(np.sqrt(2*np.pi)*kernparam))*np.exp(-.5*(Dist**2)/(2*(kernparam**2)))
            else:
                K = np.zeros((Dist.shape[1],Dist.shape[2]))
        return K


class ETWSVM(BaseEstimator,ClassifierMixin):
    """
    Class ETWSVM: it is a sklearn estimator with parameters:
        kernfunction: str, lin -> linear kernel, rbf -> Radial Basis kernel
                            nrbf -> normalization Radial Basis Kernel. 
                            Default lin
        kernparam: only rbf or nrbf kernel, is the scale parameter
        c11: int, first regularization parameter of minority hiperplane
        c12: int, first regularization parameter of majority hiperplane
        c21: int, second regularization parameter (slack variables) of minority hiperplane
        c22: int, second regularization parameter (slack variables) of majority hiperplane
        
        epsilon: float, avoids divisional zero
        
    METHODS:
        fit(self,X,t)
        predict(self,X)

    C. Jimenez-Castaño, A. Alvarez-Meza, A. Orozco-Gutierrez,
    Enhanced automatic twin support vector machine for imbalanced data classification,
    Pattern Recognition,
    Volume 107,
    2020,
    107442,
    ISSN 0031-3203,
    https://doi.org/10.1016/j.patcog.2020.107442.
    (http://www.sciencedirect.com/science/article/pii/S0031320320302454)
    Abstract: Most of the classification approaches assume that the sample distribution among classes is balanced. Still, such an assumption leads to biased performance over the majority class. This paper proposes an enhanced automatic twin support vector machine – (EATWSVM) to deal with imbalanced data, which incorporates a kernel representation within a TWSVM-based optimization. To learn the kernel function, we impose a Gaussian similarity, ruled by a Mahalanobis distance, and couple a centered kernel alignment-based approach to improving the data separability. Besides, we suggest a suitable range to fix the regularization parameters concerning both the dataset’ imbalance ratio and overlap. Lastly, we adopt One-vs-One and One-vs-Rest frameworks to extend our EATWSVM formulation for multi-class tasks. Obtained results on synthetic and real-world datasets show that our approach outperforms state-of-the-art methods concerning classification performance and training time.
    Keywords: Imbalanced data; Kernel methods; Twin support vector machines
    """
    def __init__(self,kernfunction = 'lin',kernparam = 1,c11 = 1,c12=1,c21 =1,c22=1,epsilon = 1e-6):
        self.kernfunction = kernfunction
        self.kernparam = kernparam
        self.c11 = c11
        self.c12 = c12
        self.c21 = c22
        self.c22 = c22
        self.epsilon = epsilon
        # self.X = X
        # self.t = t
        
    
    def fit(self,X,t):
        # X = self.X
        # t = self.t
        self.y,self.labels = Targets2Labels(t)
        y = self.y
                
        kernfunction = self.kernfunction
        kernparam = self.kernparam
        # labels = self.labels
        
        
        c11 = self.c11
        c12 = self.c12
        c21 = self.c21
        c22 = self.c22
        
        if c11 == 0:
            c11 = self.epsilon
        if c12 == 0:
            c12 = self.epsilon
        self.c1 = c12
        
        
        indexPos = np.where(y == 1)[0]
        indexNeg = np.where(y == -1)[0]
        X_pos = X[indexPos][:]
        X_neg = X[indexNeg][:]
        N_pos = X_pos.shape[0]
        N_neg = X_neg.shape[0]
        # XX = np.concatenate((X_pos,X_neg))
        
        # N = X.shape[0] +1
        
        kern = kernel(kernfunction = kernfunction,kernparam = kernparam)
        # Compute kernel matrix
        K_pp = kern.ComputeKern((X_pos,X_pos)) 
        K_nn = kern.ComputeKern((X_neg,X_neg)) 
        K_pn = kern.ComputeKern((X_pos,X_neg)) 
        K_np = K_pn.T
        # Compute extended kernel matrix
        Ke_pp = K_pp + np.ones((N_pos,N_pos))
        Ke_nn = K_nn + np.ones((N_neg,N_neg))
        Ke_pn = K_pn + np.ones((N_pos,N_neg))
        Ke_np = Ke_pn.T
        
        A_pos =np.linalg.inv(Ke_pp + c11*np.eye(N_pos))
        A_neg =np.linalg.inv(Ke_nn + c12*np.eye(N_neg))
        
        H_pos = (1/c11)*(Ke_nn - np.dot(Ke_np,np.dot(A_pos,Ke_pn)))
        H_pos = .5*(H_pos.T + H_pos) #+ np.eye(H_pos.shape[0])*self.epsilon
        H_neg = (1/c12)*(Ke_pp - np.dot(Ke_pn,np.dot(A_neg,Ke_np)))
        H_neg = .5*(H_neg.T + H_neg) #+ np.eye(H_neg.shape[0])*self.epsilon
        u_pos = -1*np.ones((N_neg,1)).reshape((N_neg,))
        u_neg = -1*np.ones((N_pos,1)).reshape((N_pos,))
        
        if type(c21) == float or type(c21) == int:
            C21 = (1/(N_neg*c21))*np.ones((N_neg,1))
        if type(c22) == float or type(c22)==int:
            C22 = (1/(N_pos*c22))*np.ones((N_pos,1))
        G_pos = np.concatenate((-1*np.eye(N_neg),np.eye(N_neg)))
        G_neg = np.concatenate((-1*np.eye(N_pos),np.eye(N_pos)))
        h_pos = np.concatenate((np.zeros((N_neg,1)),C21)).reshape((2*N_neg,))
        h_neg = np.concatenate((np.zeros((N_pos,1)),C22)).reshape((2*N_pos,))
        
        alpha_neg = np.array(solve_qp(H_pos,u_pos,G_pos,h_pos,solver = 'cvxopt'))
        alpha_neg = alpha_neg.reshape((alpha_neg.size,1))

        alpha_pos = np.array(solve_qp(H_neg,u_neg,G_neg,h_neg,solver = 'cvxopt'))
        alpha_pos = alpha_pos.reshape((alpha_pos.size,1))
        
        Mpos = K_nn - 2*np.dot(K_np,np.dot(A_pos,Ke_pn)) + np.dot(Ke_np,np.dot(A_pos,np.dot(K_pp,np.dot(A_pos,Ke_pn))))
        Mneg = K_pp - 2*np.dot(K_pn,np.dot(A_neg,Ke_np)) + np.dot(Ke_pn,np.dot(A_neg,np.dot(K_nn,np.dot(A_neg,Ke_np))))
        norm_W_pos = np.sqrt((1/(c11**2))*np.dot(alpha_neg.T,np.dot(Mpos,alpha_neg)))
        norm_W_neg = np.sqrt((1/(c12**2))*np.dot(alpha_pos.T,np.dot(Mneg,alpha_pos)))
      
        if norm_W_pos == 0:
            norm_W_pos = self.epsilon
        if norm_W_neg == 0:
            norm_W_neg = self.epsilon
        
        self.norm_w_pos = norm_W_pos
        self.norm_w_neg = norm_W_neg
        self.Ke_pp = Ke_pp
        self.Ke_nn = Ke_nn
        self.Ke_pn = Ke_pn
        self.Ke_np = Ke_np
        self.X_pos = X_pos
        self.X_neg = X_neg
        self.A_pos = A_pos
        self.A_neg = A_neg
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.kern = kern
        return self

    def predict(self,Xtest):
        
        kern = self.kern
        labels = self.labels
        c11 = self.c11
        c12 = self.c12
        # Ke_pp = self.Ke_pp
        Ke_pn = self.Ke_pn
        # Ke_nn = self.Ke_nn
        Ke_np = self.Ke_np
        X_pos = self.X_pos
        X_neg = self.X_neg
        A_pos = self.A_pos
        A_neg = self.A_neg
        alpha_pos = self.alpha_pos
        alpha_neg = self.alpha_neg
        norm_w_pos = self.norm_w_pos
        
        norm_w_neg = self.norm_w_neg
        
        ntest = Xtest.shape[0]
        
        Ktest_neg = kern.ComputeKern((Xtest,X_neg)) + np.ones((ntest,X_neg.shape[0]))
        
        Ktest_pos = kern.ComputeKern((Xtest,X_pos)) + np.ones((ntest,X_pos.shape[0]))
        
        prod_pos = Ktest_neg - np.dot(Ktest_pos,np.dot(A_pos,Ke_pn))
        prod_neg = Ktest_pos - np.dot(Ktest_neg,np.dot(A_neg,Ke_np))
        f_pos = (-1/c11)*np.dot(prod_pos,alpha_neg)
        f_neg = (1/c12)*np.dot(prod_neg,alpha_pos)
        
        d2pos = np.abs(f_pos)/norm_w_pos
        d2neg = np.abs(f_neg)/norm_w_neg
        
        D = d2neg - d2pos
        
        y_est = np.sign(D)
        Nt = Xtest.shape[0]
        
        t_est = []
        
        for i in range(Nt):
            if y_est[i] == 1 or y_est[i] == 0:
                t_est.append(labels[0])
            elif y_est[i] == -1:
                t_est.append(labels[1])
            
        t_est = np.array(t_est)
        return t_est

class EATWSVM(BaseEstimator,ClassifierMixin):
    """
    Class ETWSVM: it is a sklearn estimator with parameters:
        kernfunction: str, lin -> linear kernel, rbf -> Radial Basis kernel
                            nrbf -> normalization Radial Basis Kernel. 
                            Default lin
        kernparam: only rbf or nrbf kernel, is the scale parameter
        c11: int, first regularization parameter of minority hiperplane
        c12: int, first regularization parameter of majority hiperplane
        c21: int, second regularization parameter (slack variables) of minority hiperplane
        c22: int, second regularization parameter (slack variables) of majority hiperplane
        
        epsilon: float, avoids divisional zero
        
    METHODS:
        fit(self,X,t)
        predict(self,X)
    ATRIBUTES:
    

    C. Jimenez-Castaño, A. Alvarez-Meza, A. Orozco-Gutierrez,
    Enhanced automatic twin support vector machine for imbalanced data classification,
    Pattern Recognition,
    Volume 107,
    2020,
    107442,
    ISSN 0031-3203,
    https://doi.org/10.1016/j.patcog.2020.107442.
    (http://www.sciencedirect.com/science/article/pii/S0031320320302454)
    Abstract: Most of the classification approaches assume that the sample distribution among classes is balanced. Still, such an assumption leads to biased performance over the majority class. This paper proposes an enhanced automatic twin support vector machine – (EATWSVM) to deal with imbalanced data, which incorporates a kernel representation within a TWSVM-based optimization. To learn the kernel function, we impose a Gaussian similarity, ruled by a Mahalanobis distance, and couple a centered kernel alignment-based approach to improving the data separability. Besides, we suggest a suitable range to fix the regularization parameters concerning both the dataset’ imbalance ratio and overlap. Lastly, we adopt One-vs-One and One-vs-Rest frameworks to extend our EATWSVM formulation for multi-class tasks. Obtained results on synthetic and real-world datasets show that our approach outperforms state-of-the-art methods concerning classification performance and training time.
    Keywords: Imbalanced data; Kernel methods; Twin support vector machines
    """
    def __init__(self,kernfunction = 'lin',kernparam = 1,c11 = 1,c12 =1,c21=1,c22=1,epsilon = 1e-6,Q = 0.98):
        self.kernfunction = kernfunction
        self.kernparam = kernparam
        self.c11 = c11
        self.c12 = c12
        self.c21 = c21
        self.c22 = c22
        self.epsilon = epsilon
        self.Q = Q
        # self.X = X
        # self.t = t
        
    
    def fit(self,X,t):
        # X = self.X
        # t = self.t
        self.y,self.labels = Targets2Labels(t)
        y = self.y
                
        kernfunction = self.kernfunction
        kernparam = self.kernparam
        # labels = self.labels
        
        cka = CKA(showCommandLine = 0, Q = self.Q)
        cka.fit(X,y)
        
        
        
        c11 = self.c11
        c12 = self.c12
        c21 = self.c21
        c22 = self.c22
        
        if c11 == 0:
            c11 = self.epsilon
        if c12 == 0:
            c12 = self.epsilon
        
        
        indexPos = np.where(y == 1)[0]
        indexNeg = np.where(y == -1)[0]
        X_pos = X[indexPos][:]
        X_neg = X[indexNeg][:]
        N_pos = X_pos.shape[0]
        N_neg = X_neg.shape[0]
        # XX = np.concatenate((X_pos,X_neg))
        
        # N = X.shape[0] +1
        
        kern = kernel(kernfunction = kernfunction,kernparam = kernparam,E = cka.Wcka)
        # Compute kernel matrix
        K_pp = kern.ComputeKern((X_pos,X_pos)) 
        K_nn = kern.ComputeKern((X_neg,X_neg)) 
        K_pn = kern.ComputeKern((X_pos,X_neg)) 
        K_np = K_pn.T
        # Compute extended kernel matrix
        Ke_pp = K_pp + np.ones((N_pos,N_pos))
        Ke_nn = K_nn + np.ones((N_neg,N_neg))
        Ke_pn = K_pn + np.ones((N_pos,N_neg))
        Ke_np = Ke_pn.T
        
        A_pos =np.linalg.inv(Ke_pp + c11*np.eye(N_pos))
        A_neg =np.linalg.inv(Ke_nn + c12*np.eye(N_neg))
        
        H_pos = (1/c11)*(Ke_nn - np.dot(Ke_np,np.dot(A_pos,Ke_pn)))
        H_pos = .5*(H_pos.T + H_pos) #+ np.eye(H_pos.shape[0])*self.epsilon
        H_neg = (1/c12)*(Ke_pp - np.dot(Ke_pn,np.dot(A_neg,Ke_np)))
        H_neg = .5*(H_neg.T + H_neg) #+ np.eye(H_neg.shape[0])*self.epsilon
        u_pos = -1*np.ones((N_neg,1)).reshape((N_neg,))
        u_neg = -1*np.ones((N_pos,1)).reshape((N_pos,))
        
        if type(c21) == float or type(c21) == int:
            C21 = (1/(N_neg*c21))*np.ones((N_neg,1))
        if type(c22) == float or type(c22)==int:
            C22 = (1/(N_pos*c22))*np.ones((N_pos,1))
        G_pos = np.concatenate((-1*np.eye(N_neg),np.eye(N_neg)))
        G_neg = np.concatenate((-1*np.eye(N_pos),np.eye(N_pos)))
        h_pos = np.concatenate((np.zeros((N_neg,1)),C21)).reshape((2*N_neg,))
        h_neg = np.concatenate((np.zeros((N_pos,1)),C22)).reshape((2*N_pos,))
        
        alpha_neg = np.array(solve_qp(H_pos,u_pos,G_pos,h_pos,solver = 'cvxopt'))
        alpha_neg = alpha_neg.reshape((alpha_neg.size,1))

        alpha_pos = np.array(solve_qp(H_neg,u_neg,G_neg,h_neg,solver = 'cvxopt'))
        alpha_pos = alpha_pos.reshape((alpha_pos.size,1))
        
        Mpos = K_nn - 2*np.dot(K_np,np.dot(A_pos,Ke_pn)) + np.dot(Ke_np,np.dot(A_pos,np.dot(K_pp,np.dot(A_pos,Ke_pn))))
        Mneg = K_pp - 2*np.dot(K_pn,np.dot(A_neg,Ke_np)) + np.dot(Ke_pn,np.dot(A_neg,np.dot(K_nn,np.dot(A_neg,Ke_np))))
        norm_W_pos = np.sqrt((1/(c11**2))*np.dot(alpha_neg.T,np.dot(Mpos,alpha_neg)))
        norm_W_neg = np.sqrt((1/(c12**2))*np.dot(alpha_pos.T,np.dot(Mneg,alpha_pos)))
      
        if norm_W_pos == 0:
            norm_W_pos = self.epsilon
        if norm_W_neg == 0:
            norm_W_neg = self.epsilon
        
        self.norm_w_pos = norm_W_pos
        self.norm_w_neg = norm_W_neg
        self.Ke_pp = Ke_pp
        self.Ke_nn = Ke_nn
        self.Ke_pn = Ke_pn
        self.Ke_np = Ke_np
        self.X_pos = X_pos
        self.X_neg = X_neg
        self.A_pos = A_pos
        self.A_neg = A_neg
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.kern = kern
        return self

    def predict(self,Xtest):
        
        kern = self.kern
        labels = self.labels
        c11 = self.c11
        c12 = self.c12
        # Ke_pp = self.Ke_pp
        Ke_pn = self.Ke_pn
        # Ke_nn = self.Ke_nn
        Ke_np = self.Ke_np
        X_pos = self.X_pos
        X_neg = self.X_neg
        A_pos = self.A_pos
        A_neg = self.A_neg
        alpha_pos = self.alpha_pos
        alpha_neg = self.alpha_neg
        norm_w_pos = self.norm_w_pos
        
        norm_w_neg = self.norm_w_neg
        
        ntest = Xtest.shape[0]
        
        Ktest_neg = kern.ComputeKern((Xtest,X_neg)) + np.ones((ntest,X_neg.shape[0]))
        
        Ktest_pos = kern.ComputeKern((Xtest,X_pos)) + np.ones((ntest,X_pos.shape[0]))
        
        prod_pos = Ktest_neg - np.dot(Ktest_pos,np.dot(A_pos,Ke_pn))
        prod_neg = Ktest_pos - np.dot(Ktest_neg,np.dot(A_neg,Ke_np))
        f_pos = (-1/c11)*np.dot(prod_pos,alpha_neg)
        f_neg = (1/c12)*np.dot(prod_neg,alpha_pos)
        
        d2pos = np.abs(f_pos)/norm_w_pos
        d2neg = np.abs(f_neg)/norm_w_neg
        
        D = d2neg - d2pos
        
        y_est = np.sign(D)
        Nt = Xtest.shape[0]
        
        t_est = []
        
        for i in range(Nt):
            if y_est[i] == 1 or y_est[i] == 0:
                t_est.append(labels[0])
            elif y_est[i] == -1:
                t_est.append(labels[1])
            
        t_est = np.array(t_est)
        return t_est

# def main():
#%% ********************************* impotr data *********************************

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X,y = load_iris(return_X_y=True)
y[y==2] = 1 # change Binary classification
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)


clf = EATWSVM(kernfunction='rbf')


clf.fit(X_train,y_train)

y_est = clf.predict(X_test)

print(classification_report(y_test,y_est))
