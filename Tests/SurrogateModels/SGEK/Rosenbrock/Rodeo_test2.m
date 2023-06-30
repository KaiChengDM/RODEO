
clc;  clear;

format long;

n = 10;

 if n > 2
  g = @(x)sum((1-x(:,1:n-1)).^2')' + sum(100.*(x(:,2:n)-x(:,1:n-1).^2).^2')';
 else
  g = @(x)((1-x(:,1:n-1)).^2')' + (100.*(x(:,2:n)-x(:,1:n-1).^2).^2')';
 end

 for i = 1:n-1
  pd{i} = @(x) -2.*(1-x(:,i)) - 400.*(x(:,i+1)-x(:,i).^2).*x(:,i);
 end

pd{n} = @(x) 200.*(x(:,n)-x(:,n-1).^2);

const{1} = @(x)sum(x')'-10;

for i = 1 : n
  pd_const{1,i} = @(x)1;
end

lb = -5.*ones(1,n);  ub = 5.*ones(1,n); N = 100; N1 = 1000;

% pp = sobolset(n,'Skip',5); u=net(pp,N);  
% pp1 = sobolset(n,'Skip',1001,'Leap',N1); u1=net(pp1,N1);  

sig = ones(1,n); mu = zeros(1,n);
u = normcdf(lhsnorm(mu,diag(sig.^2),N));
u1 = normcdf(lhsnorm(mu,diag(sig.^2),N1));

for i=1:n
  x(:,i) = u(:,i)*(ub(i)-lb(i))+lb(i);
  xtest(:,i)=u1(:,i)*(ub(i)-lb(i))+lb(i);
end

y_obj = g(x); y1 = g(xtest);


train_data = [x y_obj];

test_data = [xtest y1];

% load('trainingData1.csv')

% load('trainingData.csv')
% 
% trainingData = trainingData1(1:50,:);

writematrix(train_data,'trainingData.csv')

writematrix(test_data,'testDataInput.csv')

writematrix(xtest,'testDataInput.csv')
writematrix(y1,'testDataOutput.csv')


% 
% 
% load('surrogateTest.csv')
% 
% X = trainingData(:,1:2); Y = trainingData(:,3);
% 
% lb = min(X); ub =  max(X);
% 
% xtest = surrogateTest(:,1:2); y1 = surrogateTest(:,3);
% 
% N = 50; 
% 
% x = X(1:N,:); y = Y(1:N);
%  
% n = 2; 

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 5*10^-4.*ones(1,n);
hyperpar.ub = 5*ones(1,n);
hyperpar.corr_fun      = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 10;

inputpar.lb = lb;
inputpar.ub = ub;

inputpar.x = x;
inputpar.y = y_obj;

t1=clock;
  Kriging_Model=Kriging_fit(inputpar,hyperpar);
t2=clock;


[Mean Variance]= Kriging_predictor(xtest,Kriging_Model);
MSE=mean((Mean-y1).^2)/var(y1)
