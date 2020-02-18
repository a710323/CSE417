% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)
load zip.train;

fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y_train = subsample(:,1);
X_train = subsample(:,2:257);
ct = fitctree(X_train,Y_train,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
bee = BaggedTrees(X_train, Y_train, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee(200));


fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y_train = subsample(:,1);
X_train = subsample(:,2:257);
ct = fitctree(X_train,Y_train,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
bee = BaggedTrees(X_train, Y_train, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee(200));