function [first, Minimum, ranPick] = coinSimulation(n, num_exper)
first = [num_exper];
Minimum = [num_exper];
ranPick = [num_exper];

for m = 1:num_exper
    for k = 1:n
        coin = sum(randi([0:1],[n,1000]));
        %averageAll = sum(coin)/1000
        %average3 = (Minimum + first + ranPick) / 3
    end
    first(m,1) = coin(1)/10;
    Minimum(m,1) = min(coin)/10;
    ranPick(m,1) = coin(randperm(length(coin),1))/10;
end
