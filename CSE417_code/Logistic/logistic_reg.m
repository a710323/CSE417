function [ w, e_in,num_its, execution_time] = logistic_reg( X, y, w_init, max_its, eta )
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix (without an initial column of 1s)
%       y : data labels (plus or minus 1)
%       w_init: initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta: learning rate
    
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (as defined in LFD)

X = [ones(size(X,1),1) X];
w = w_init;
tic
for num_its = 0 : max_its
    g = zeros(size(X));
    for i = 1 : size(X,1)
        g(i,:) = y(i).*X(i,:) / (1 + exp(y(i)*(w'*X(i,:)')));
    end
    gradient = -mean(g);
    v = -gradient';
    w = w + eta*(mean(g)');
    if( all( abs(gradient) < 10^-6) )
        break
    end
end
e_in = mean( log( 1 + exp(-y.*(X*w) ) ) );
execution_time = toc;
end