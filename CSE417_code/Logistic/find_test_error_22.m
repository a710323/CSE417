function [test_error] = find_test_error(w, X, y)

% find_test_error: compute the test error of a linear classifier w. The
%  hypothesis is assumed to be of the form sign([1 x(n,:)] * w)
%  Inputs:
%		w: weight vector
%       X: data matrix (without an initial column of 1s)
%       y: data labels (plus or minus 1)
%     
%  Outputs:
%        test_error: binary error of w on the data set (X, y) error; 
%        this should be between 0 and 1. 

%test(a):
%[test_error] = find_test_error(w, D_train(:,1:end-1), D_train(:,end))
%[test_error] = find_test_error(w, D_test(:,1:end-1), D_test(:,end))
%test(b):
%[test_error] = find_test_error(w, D_test_norm(:,1:end-1), D_test_norm(:,end))

%add ones
ones_vec = ones(size(X,1), 1);
X = horzcat(ones_vec, X);

N = size(X, 1);

tic;
%calculate actual
y_val = X * w;
y_val (y_val < 0.5) = -1;
y_val (y_val >= 0.5) = 1;
%calculate predicted
h = sign(X * w);
%get count of errors: (actual != predicted)
num_rows = size(y_val, 1);
count = num_rows - sum(y_val == h);
toc;

%return output
test_error = count ./ N;
end

