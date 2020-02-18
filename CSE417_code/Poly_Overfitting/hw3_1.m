data_train = csvread('cleveland_train.csv',1,0);
X_train = data_train(:,1:end-1);
y_train = data_train(:,end);
y_train = y_train - (y_train == 0);
y_train_b = data_train(:,end);

data_test = csvread('cleveland_test.csv', 1,0 );
X_test = data_test(:,1:end - 1);
y_test = data_test(:,end);
y_test = y_test - (y_test==0);%converting y_test from {0,1} to {-1,1}

Z_train = zscore(X_train);
Z_test = zscore(X_test);

w_init = zeros(size(X_train,2)+1,1);

display('max_its = 10k')
eta = 10^-5;
max_its = 10^4;
tolerance = 10^-3;
[ w, e_in, num_its, execution_time] = logistic_reg( X_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, X_train, y_train );
[ test_error_test] = find_test_error( w, X_test, y_test );
result_10k = [e_in; num_its; execution_time; clss_error_train; test_error_test];

display('max_its = 100k')
max_its = 10^5;
[ w, e_in, num_its, execution_time] = logistic_reg( X_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, X_train, y_train );
[ test_error_test] = find_test_error( w, X_test, y_test );
result_100k = [e_in; num_its; execution_time; clss_error_train; test_error_test];

display('max_its = 1 million')
max_its = 10^6;
[ w, e_in, num_its, execution_time] = logistic_reg( X_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, X_train, y_train );
[ test_error_test] = find_test_error( w, X_test, y_test );
result_1m = [e_in; num_its; execution_time; clss_error_train; test_error_test];

%part(2): compare the best-achieved result above with function glmfit
display('glmfit')
tic
w = glmfit(X_train,y_train_b,'binomial');
execution_time = toc;
[ clss_error_train] = find_test_error( w, X_train, y_train );
[ test_error_test] = find_test_error( w, X_test, y_test );
result_glmfit = [execution_time; clss_error_train; test_error_test];

%part(3): scaling x to zscore(x), different eta and tolerance = 10^-6. Run
%until each element in gradient below tolerance
max_its = 10^5;
tolerance = 10^-6;
display('eta = 10^-5')
eta = 10^-5; 
[ w, e_in, num_its, execution_time] = logistic_reg( Z_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, Z_train, y_train );
[ test_error_test] = find_test_error( w, Z_test, y_test );
result_eta_10n5 = [e_in; num_its; execution_time; clss_error_train; test_error_test];

display('eta = 10^-4')
eta = 10^-4; 
[ w, e_in, num_its, execution_time] = logistic_reg( Z_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, Z_train, y_train );
[ test_error_test] = find_test_error( w, Z_test, y_test );
result_eta_10n4 = [e_in; num_its; execution_time; clss_error_train; test_error_test];

display('eta = 10^-3')
eta = 10^-3; 
[ w, e_in, num_its, execution_time] = logistic_reg( Z_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, Z_train, y_train );
[ test_error_test] = find_test_error( w, Z_test, y_test );
result_eta_10n3 = [e_in; num_its; execution_time; clss_error_train; test_error_test];

display('eta = 10^-2')
eta = 10^-2; 
[ w, e_in, num_its, execution_time] = logistic_reg( Z_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, Z_train, y_train );
[ test_error_test] = find_test_error( w, Z_test, y_test );
result_eta_10n2 = [e_in; num_its; execution_time; clss_error_train; test_error_test];

display('eta = 0.1')
eta = 0.1; 
[ w, e_in, num_its, execution_time] = logistic_reg( Z_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, Z_train, y_train );
[ test_error_test] = find_test_error( w, Z_test, y_test );
result_eta_10n1 = [e_in; num_its; execution_time; clss_error_train; test_error_test];

display('eta = 1')
eta = 1; 
[ w, e_in, num_its, execution_time] = logistic_reg( Z_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, Z_train, y_train );
[ test_error_test] = find_test_error( w, Z_test, y_test );
result_eta_1 = [e_in; num_its; execution_time; clss_error_train; test_error_test];

display('eta = 3')
eta = 3; 
[ w, e_in, num_its, execution_time] = logistic_reg( Z_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, Z_train, y_train );
[ test_error_test] = find_test_error( w, Z_test, y_test );
result_eta_3 = [e_in; num_its; execution_time; clss_error_train; test_error_test];

display('eta = 5')
eta = 5; 
[ w, e_in, num_its, execution_time] = logistic_reg( Z_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, Z_train, y_train );
[ test_error_test] = find_test_error( w, Z_test, y_test );
result_eta_5 = [e_in; num_its; execution_time; clss_error_train; test_error_test];

display('eta = 7')
eta = 7; 
[ w, e_in, num_its, execution_time] = logistic_reg( Z_train, y_train, w_init, max_its, eta);
[ clss_error_train] = find_test_error( w, Z_train, y_train );
[ test_error_test] = find_test_error( w, Z_test, y_test );
result_eta_7 = [e_in; num_its; execution_time; clss_error_train; test_error_test];