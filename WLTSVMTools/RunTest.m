clear all;
%% Training Dataset
Data.A = rand(20,2);
Data.B = rand(100,2)+ 3;

%% Testing Dataset
Data.TestX = [rand(100,2);rand(100,2)+ 3;];
Data.TestY = [ones(100,1);-ones(100,1)];

%% Parameter setting
FunPara.p1=1;FunPara.p2=1;
FunPara.kerfPara.pars = 0.5;
FunPara.kerfPara.type = 'rbf';


%% Training WLTSVM with predetermine parameter
Predict_Y =WLTSVM(Data.TestX,Data,FunPara);

%% Calculating accuracy
Accuracy = sum(Predict_Y == Data.TestY)/length(Data.TestY)
