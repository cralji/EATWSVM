% function bestalpha=LSVMR(Q,t,C,smallvalue)
% TBSVM algorithm: An Improved TWSVM Learning Technique.
%
% Output:
%         bestu   -- the solution of the QP;
%
% Input:
%         Q    -- the input matrix
%         c1,c2   -- the parameter 
%         maxIter    -- 迭代次数
%         tol          -- Terminating;
%
% For questions, email: shaoyuanhai21@163.com

function bestu=lsvm(Q,c,itmax,tol)
%Initialization
e=ones(size(Q,1),1);
iter=0;
u=Q\e;
oldu=u+1;
beta=1.9/c;%beta是一个参数；u是拉格朗日乘子，即要求的解；
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 while iter<itmax & norm(oldu-u)>tol
      z= Q*u - e - beta*u;
      oldu=u;
%       alpha=inv(Q1)*pl;
      u=Q\((abs(z)+z)/2 + e);
      iter=iter+1;
  end;
 % opt=norm(u-oldu);
   bestu=u;
  %[iter opt]
   function pl = pl(x);  
           pl = (x+abs(x))/2;
   return;
