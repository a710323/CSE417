function [v] = coinSimulation2(num_exper)
t = [num_exper,2]
for m = 1:num_exper

    coin = sum(randi([0:1],[6,2]));
        %averageAll = sum(coin)/1000
        %average3 = (Minimum + first + ranPick) / 3
    t(m,1:2) = coin/6;
end
v = sum(t)/num_exper
end
