load zip.test;

fprintf('Working on the One-vs-Three problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y_test = subsample(:,1);
X_test = subsample(:,2:257);
load zip.train;

fprintf('Working on the Three-vs-Five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y_train = subsample(:,1);
X_train = subsample(:,2:257);

bee1 = BaggedTrees_test(X_train, Y_train, X_test, Y_test, 1);
fprintf('The test error of one decision trees is %.4f\n', bee1);
bee = BaggedTrees_test(X_train, Y_train,X_test, Y_test, 200);
fprintf('The test error of 200 bagged decision trees is %.4f\n', bee(200));