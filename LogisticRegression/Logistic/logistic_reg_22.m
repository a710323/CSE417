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

%test(a): 
%[t, w, e_in] = logistic_reg(D_train(:,1:end-1), D_train(:,end), w_init_train, 10^4, 10^(-5))
%[t, w, e_in] = logistic_reg(D_test(:,1:end-1), D_test(:,end), w_init_test, 10^4, 10^(-5))
%test(b):
%[t, w, e_in] = logistic_reg(D_train_norm(:,1:end-1), D_train_norm(:,end), w_init_train, inf, 0.01)
%[t, w, e_in] = logistic_reg(D_test_norm(:,1:end-1), D_test_norm(:,end), w_init_test, inf, 0.01)

%add ones
ones_vec = ones(size(X,1), 1);
X = horzcat(ones_vec, X);

w = w_init;
N = size(X, 1);
t = 0;
gt = inf;

tic;
% while(t < max_its && any(abs(gt) >= 10^(-3)))
while(t < max_its && any(abs(gt) >= 10^(-6)))
    %compute gradient
%     sizeX = size(X)
%     sizey = size(y)
%     sizew = size(w)
    gt = sum(X.*y ./ (1 + exp(X*w.*y)))./ (-N);
    
    %track num iter of gradient descent
    t = t + 1;
    
    %set direction to move
    vt = -1 .* gt;
    
    %update weights
    w = w + (eta .* vt)';
end
toc;

%calculate cross-entropy error
e_in = sum(log(1 + exp(X*w.*(-1.*y))))/N;

end

