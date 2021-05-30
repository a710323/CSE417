function [w, iterations] = perceptron_learn( data_in )
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for
% data_in = 

% obtain the number of rows and columns from data_in

[m,n] = size(data_in);

% initilize some variables
iterations = 0;
x = data_in(1:m-1,1:n);
y = data_in(m, 1:n);
w = zeros([m-1 1]);
% sum of y should equal to number of data
while sum(sign(w' * x) == y) ~= n
    for k = 1:n
        h = sign(w' * x);
        if h(k) ~= y(k)
            w = w + x(1:end,k)*y(k);
            iterations = iterations + 1;    
            break
        end
%     a = (sign(w'*c) ~= d);
%     k = find(a);
%     j = randi(length(k));
    end
end
end

