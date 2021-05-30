function [ oobErr ] = BaggedTrees( X, Y, numBags )
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

[N,d] = size(X);
a = unique(Y);
Arr = [];
record = zeros(N, 1);
for m = 1:numBags
    i = datasample(1:N,N); %generate numbers from 1 to N,replace.
    x = X(i,:);     %generate training set T
    y = Y(i,:);
    model = fitctree(x,y);
    oobIndex = setdiff(1:N, unique(i));
    outbagdata = X(oobIndex,:);
    Arr = [Arr; i];
    UniArr = unique(Arr);
    UniLeng = length(UniArr);
    Yout = predict(model,outbagdata);
    Yout(Yout==a(1)) = 1;
    Yout(Yout==a(2)) = -1;
    
    record(oobIndex) = record(oobIndex) + Yout;
    Yoob = record(UniArr);
    Yoob(Yoob>=0) = a(1);
    Yoob(Yoob<0) = a(2);
    error = sum((abs(Y(UniArr)-Yoob))/2);
    oobErr(m) = error/UniLeng;
end



plot(oobErr);
xlabel('number of bag');
ylabel('out of bag error');
title('numBags=200, Compare 3 and 5');
saveas(gcf, "h5_2.png")
end
