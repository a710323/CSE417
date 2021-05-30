function [ oobErrTest ] = BaggedTrees_test( X_train, Y_train, X_test, Y_test, numBags )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X : Matrix of training data
%       Y : Vector of classes of the training examples
%       numBags : Number of trees to learn in the ensemble
%
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function

[N,d] = size(X_train);
[P,Q] = size(X_test);
a = unique(Y_test);
record = zeros(P, 1);
for m = 1:numBags
    i = datasample(1:N,N); %generate numbers from 1 to N,replace.
    x = X_train(i,:);     %generate training set T
    y = Y_train(i,:);
    traintree = fitctree(x,y);
    outbagdata = X_test;
    Yout = predict(traintree,outbagdata);
    Yout(Yout==a(1))= 1;
    Yout(Yout==a(2))= -1;
    record = record + Yout;

    Yoob = record;
    Yoob(Yoob>=0) = a(1);
    Yoob(Yoob<0) = a(2);
    error = sum(abs(Y_test - Yoob)/2);
    oobErrTest(m)= error/P;
end



plot(oobErrTest);
xlabel('number of bag');
ylabel('Test Error');
title('numBags=200, Compare 3 and 5');
saveas(gcf, "h5_4.png")
end
