function [ train_set, test_set ] = generate_dataset_hu( Q_f, N_train, N_test, sigma )
%GENERATE_DATASET Generate training and test sets for the Legendre
%polynomials example
%   Inputs:
%       Q_f: order of the hypothesis
%       N_train: number of training examples
%       N_test: number of test examples
%       sigma: standard deviation of the stochastic noise
%   Outputs:
%       train_set and test_set are both 2-column matrices in which each row
%       represents an (x,y) pair

%generate training data set
train_x = unifrnd(-1,1,N_train,1);
a_in = normrnd(0,1,Q_f+1,1);
norm_coef = 0;
for i = 1:(Q_f+1)
    norm_coef = norm_coef + 1/(2*(i-1)+1);
end
a_in = a_in/sqrt(norm_coef);

L_x = computeLegPoly_hu(train_x,Q_f);
train_y = (a_in'*L_x)';
noise = normrnd(0,1,N_train,1);
train_y = train_y + sigma*noise;

train_set = [train_x, train_y];


%generate test data set
test_x = unifrnd(-1,1,N_test,1);
a_out = normrnd(0,1,Q_f+1,1);
norm_coef = 0;
for i = 1:(Q_f+1)
    norm_coef = norm_coef + 1/(2*(i-1)+1);
end
a_out = a_out/sqrt(norm_coef);

L_x = computeLegPoly_hu(test_x,Q_f);
test_y = (a_out'*L_x)';
noise = normrnd(0,1,N_test,1);
test_y = test_y + sigma*noise;

test_set = [test_x, test_y];

end
