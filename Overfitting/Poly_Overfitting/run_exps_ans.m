train_data = readmatrix('cleveland_train.csv');
train_data(train_data(:,end) == 0,end) = -1;
test_data = readmatrix('cleveland_test.csv');
test_data(test_data(:,end) == 0,end) = -1;

train_X = train_data(:,1:end-1);
train_y = train_data(:,end);
test_X = test_data(:,1:end-1);
test_y = test_data(:,end);

d = size(train_X,2);
w_init = zeros(d+1,1);

output_a = zeros(3,4);
for i = 1:3
    tic;
    [t, w, output_a(i,1)] = logistic_reg_student(train_X, train_y, w_init, 10^(i+3), 10^(-5));
    output_a(i,4) = toc;
 
    output_a(i,2) = find_test_error_answer_ans(w, train_X, train_y);
    output_a(i,3) = find_test_error_answer_ans(w, test_X, test_y);
end

output_a

%%

train_means = mean(train_X);
train_stds = sqrt(var(train_X));
train_X = (train_X-train_means)./train_stds;
test_X = (test_X-train_means)./train_stds;

etas = [0.01, 0.1, 1, 4, 5, 6, 7, 7.5, 7.6, 7.7];
num_etas = length(etas);

output_b = zeros(num_etas,4);

for i = 1:num_etas
    tic;
    [output_b(i,1), w, output_b(i,2)] = logistic_reg_student(train_X, train_y, w_init, Inf, etas(i));
   
    output_b(i,3) = find_test_error_answer_ans(w, test_X, test_y);
    output_b(i,4) = toc;
end

output_b