%grab data from csv files
D_train = readmatrix('cleveland_train.csv');
D_test = readmatrix('cleveland_test.csv');

%remove the NaN top row
D_train = D_train(2:end, :);
D_test = D_test(2:end, :);

% change data labels to -1/+1
y_train = D_train(:, end);
y_train (y_train == 0) = -1;
D_train(:, end) = y_train;

y_test = D_test(:, end);
y_test (y_test == 0) = -1;
D_test(:, end) = y_test;

%initialize vector to zeroes
w_init_test = zeros(size(D_test,2), 1); 
w_init_train = zeros(size(D_train,2), 1); 
%^^ size = sum(x columns) - y column + initial 1

%%
%scale feature/normalize 
D_train_norm = zscore(D_train);

%get means and std dev of each column of training data
means = mean(D_train_norm);
std_devs = std(D_train_norm);

%get mean and std dev matrices to manipulate testing data
test_rows = size(D_test, 1);
norm_mean = repmat(means, test_rows, 1);
norm_std_dev = repmat(std_devs, test_rows, 1);

%normalize test data
D_test_norm = (D_test - norm_mean) ./ norm_std_dev;
