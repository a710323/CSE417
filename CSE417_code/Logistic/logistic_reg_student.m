function [t, w, e_in] = logistic_reg(X, y, w_init, max_its, eta)

% logistic_reg: learn a logistic regression model using gradient descent
%  Inputs:
%       X:       data matrix (without an initial column of 1s)
%       y:       data labels (plus or minus 1)
%       w_init:  initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta:     learning rate
%     
%  Outputs:
%        t:    the number of iterations gradient descent ran for
%        w:    learned weight vector
%        e_in: in-sample (cross-entropy) error

X_rows = size(X,1);
X = horzcat(ones(X_rows,1),X);% concatenate column of all 1s to the start
t = 0; % iterations
maxMag = intmax; % maximum gradient magnitude
w = w_init; %set our weights to initial weights
noMax = max_its == Inf;
while((noMax && maxMag>10^-6) || (t<max_its && maxMag>10^-3))% while we haven't reached max its or one of our gradients > 10^-3
    gradient = -sum((y.*X)./(1+exp(y.*((X*w)))))/X_rows;
    V = -gradient;
    w = w + eta * V;
    maxMag = max(gradient);
    t = t + 1;
end
e_in = sum(log(1+exp(-y.*((X*w)))))/X_rows;




