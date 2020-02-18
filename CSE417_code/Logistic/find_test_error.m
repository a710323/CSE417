function [ test_error ] = find_test_error( w, X, y )
%FIND_TEST_ERROR Find the test error of a linear separator
%   This function takes as inputs the weight vector representing a linear
%   separator (w), the test examples in matrix form with each row
%   representing an example (X), and the labels for the test data as a
%   column vector (y). X does not have a column of 1s as input, so that 
%   should be added. The labels are assumed to be plus or minus one. 
%   The function returns the error on the test examples as a fraction. The
%   hypothesis is assumed to be of the form (sign ( [1 x(n,:)] * w )

%[nrows, ncols] = size(X);
%x_matrix = [ones(nrows,1),X];
%y_hat = sign(x_matrix*w);
%test_error = mean(y_hat==y);

probability_threshold = 0.5;
X_ = [ones(size(X,1),1) X];
logistic_function = 1 ./ (1 + exp(-X_*w));
h_logistic = (logistic_function >= probability_threshold) - ((logistic_function >= probability_threshold)==0);
test_error = mean( y ~= h_logistic );

end