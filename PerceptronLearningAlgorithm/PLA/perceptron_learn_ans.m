function [ w iterations ] = perceptron_learn_ans( data_in )

x = data_in( : , 1:(end-1));
y = data_in( : , end);
num_iters = 0;

[num, dim] = size(x);
learning_w = zeros([dim 1])';
ready = 0;

while not(ready)
    ready = 1;
    for n = 1:num
        h = sign(learning_w * x');
        if h(n)~=y(n)
            weight = learning_w + x(n, :) * y(n);
            ready = 0;
        end
    end
    learning_w = weight;
    num_iters = num_iters + 1;
end
w = learning_w;
iterations = num_iters;


end
