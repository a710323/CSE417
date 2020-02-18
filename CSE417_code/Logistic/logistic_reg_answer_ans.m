function [t, w, e_in] = logistic_reg_answer(X, y, w_init, max_its, eta)

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

    N = size(X, 1);
    X_adj = [ones(N,1), X];
    w = w_init;
    
    if isinf(max_its)
        t = 0;
        v = Inf;
        
        while(any(abs(v) > 0.000001))
            v = sum((-y.*X_adj)./(1+exp(y.*(X_adj*w))))'/N;
            w = w - eta * v;
            t = t + 1;
        end
    else
        for t = 1:max_its
            v = sum((-y.*X_adj)./(1+exp(y.*(X_adj*w))))'/N;
            w = w - eta * v;
            if all(abs(v) < 0.001)
                break;
            end
        end
    end
    
    e_in = sum(log(1+exp(-y.*(X_adj*w))))/N;
    
end

