load zip.test;

fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y_train = subsample(:,1);
X_train = subsample(:,2:257);
ct = fitctree(X,Y,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
bee = BaggedTrees_test(X_train, Y_train,X_test, Y_test, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);