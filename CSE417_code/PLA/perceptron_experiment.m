function [num_iters bounds_minus_ni] = perceptron_experiment( N, d, num_samples )
%perceptron_experiment Code for running the perceptron experiment in HW1
%   Inputs: N is the number of training examples
%           d is the dimensionality of each example (before adding the 1)
%           num_samples is the number of times to repeat the experiment
%   Outputs: num_iters is the # of iterations PLA takes for each sample
%            bound_minus_ni is the difference between the theoretical bound
%               and the actual number of iterations
%      (both the outputs should be num_samples long)

% generate 100 data with 10 dimensional with 1 extra column
for k = 1:num_samples
    x = rand(d+1,N)*2 - 1;
    x(1,1:N) = 1;
    % generate weight vector with 11 dimension
    w_experiment = rand(d+1,1);
    w_experiment(1) = 0;
    % w'x = y(sign of each data point) +-1
    y = sign(w_experiment' * x);
    % data_in = horzcat(x,y)
    data_in = vertcat(x,y);    
    % perceptron_learn(data_in)
    [w_est,iters] = perceptron_learn(data_in);
    % store the number of iterations
    num_iters(k) = iters;
    
    % calcuate the theoretical bound
    p = min(y.*((w_experiment(1:end,1))'*x(1:end,1:end)))
    r = max(norm(x(1:end, 1:end)));
    bound(k) = (r*(norm(w_experiment))/p)^2;
    bounds_minus_ni(k) = bound(k) - num_iters(k);
end
end
