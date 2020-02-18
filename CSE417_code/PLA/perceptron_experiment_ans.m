function [ num_iters bounds_minus_ni min2_P] = perceptron_experiment_ans( N, d, num_samples )
for n = 1: num_samples
    weight = [0 rand(1,d)]';
    training = 2 * rand(d,N) -1;
    y = sign(weight(2:end,:)' * training);
    data = [training;y]';
    [learning_w, iterations] = perceptron_learn_ans(data);
    num_iters(n) = iterations;
    
    diff = sum(abs(learning_w - weight(2:end,:)'));
    max_R = max(norm(training));
    min2_P(n) = min(y .* ((weight(2:end,:)') * training))^2;
    bounds(n) = ((max_R * norm(weight))^2./min2_P)^2;
    bounds_minus_ni(n) = bounds(n) - num_iters(n);
end
end