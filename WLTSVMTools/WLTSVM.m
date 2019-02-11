function PredictY=WLTSVM(TestX,DataTrain,FunPara,sig)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% An efficient weighted Lagrangian twin support vector machine for imbalanced data classification
%
% PredictY = WLTSVM(TestX,DataTrain,FunPara)
% 
% Input:
%    TestX - Test Data matrix. Each row vector of fea is a data point.
%
%    DataTrain - Struct value in Matlab(Training data).
%                DataTrain.A: Positive input of Data matrix.
%                DataTrain.B: Negative input of Data matrix.
%
%    FunPara - Struct value in Matlab. The fields in options that can be set:
%              p1,p2: [0,inf] Paramter to tune the weight. 
%              kerfPara:Kernel parameters. See kernelfun.m.
%
% Output:
%    PredictY - Predict value of the TestX.
%
% Examples:
%    DataTrain.A = rand(100,2);
%    DataTrain.B = rand(100,2)+ 3;
%    Data.TestX = [rand(100,2);rand(100,2)+ 3;];
%    FunPara.p1=.1;
%    FunPara.p2=.1;
%    FunPara.kerfPara.type = 'lin';
%    Predict_Y =WLTSVM(TestX,DataTrain,FunPara);
% 
% Reference:
%    Yuan-Hai Shao; Wei-Jie Chen; Jing-Jing Zhang; Zhen Wang; Nai-Yang Deng,  
%    "An efficient weighted Lagrangian twin support vector machine for imbalanced
%     data classification", Submitted 2013 
%
%    Version 1.1 --Nov/2013 
%
%    Written by Yuan-Hai Shao (shaoyuanhai21@163.com) and %                Wei-Jie Chen (wjcper2008@163.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Modificación por @craljimenez
    if nargin==4
         FunPara.kerfPara.pars = sig;
    end

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic;
    Xpos = DataTrain.A;
    Xneg = DataTrain.B;
    c1 = FunPara.p1;
    c2 = FunPara.p2;
    kerfPara = FunPara.kerfPara; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Undersampling the negtive
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    K.kb = 3;K.kb1=10; 
    K.kw = 6;K.kw1 = 5;
    [Xneg2,Xneg1] = KNNSampling(Xpos,Xneg,K);
    mp1=size(Xpos,1);mn1=size(Xneg1,1);mn2=size(Xneg2,1);
    ep1=ones(mp1,1); en1=ones(mn1,1);en2=ones(mn2,1);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constructing the weight matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D1 = diag(ep1);D2 = diag(en1);
    if mp1 < mn1,D2 = mp1/mn1*D2;else D2 = mn1/mp1*D2 ;end %for plane 1
    if mp1 > mn2,D1 = mn2/mp1*D1;else D1 = mp1/mn2*D1;end % for plane 2    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constructing the kernel matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(kerfPara.type,'lin')
        H=[Xpos,ep1];
        G1=[Xneg1,en1];    
        G2=[Xneg2,en2];
    else
        X=[DataTrain.A;DataTrain.B];
        H=[kernelfun(Xpos,kerfPara,X),ep1];
        G1=[kernelfun(Xneg1,kerfPara,X),en1];     
        G2=[kernelfun(Xneg2,kerfPara,X),en2];  
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% training process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
    n=size(H,2);
    I = eye(n);
    HH1 = (c1*H'*H+I)\G1';
    GG1 = (c2*G2'*G2+I)\H';
    HH=G1*HH1+1/c1*D2;
    GG=H*GG1+1/c2*D1;
    HH = (HH +HH')/2;GG = (GG +GG')/2;
    alpha=lsvm(HH,c1,10,0.0001);
    beta=lsvm(GG,c2,10,0.0001);        
    vpos=-HH1*alpha;
    vneg=-GG1*beta;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computing w1,w2,b1,b2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
    w1=vpos(1:(length(vpos)-1));
    b1=vpos(length(vpos));
    w2=vneg(1:(length(vneg)-1));
    b2=vneg(length(vneg));
% toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    m=size(TestX,1);
    if strcmp(kerfPara.type,'lin')
        H=TestX;
        w11=sqrt(w1'*w1);
        w22=sqrt(w2'*w2);
        y1=H*w1+b1*ones(m,1);
        y2=H*w2+b2*ones(m,1);    
    else
        C=[DataTrain.A;DataTrain.B];
        H=kernelfun(TestX,kerfPara,C);
        w11=sqrt(w1'*kernelfun(X,kerfPara,C)*w1);
        w22=sqrt(w2'*kernelfun(X,kerfPara,C)*w2);
        y1=H*w1+b1*ones(m,1);
        y2=H*w2+b2*ones(m,1);
    end
    clear H; clear C;    

    mp1=y1/w11;
    mn2=y2/w22;
    PredictY = sign(abs(mn2)-abs(mp1));
end
