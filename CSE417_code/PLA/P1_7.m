
function [v] = P1_7(num_exper)
mu = 0.5;
epsilon = [0:0.01:1];
hoeffding = 2 * exp(-2*epsilon.^(2)*10);
count1 = 0;
count2 = 0;
    for k = 1:num_exper
        coin = sum(randi([0:1],[6,2]));
        v(k,[1:2]) = coin/6;
    end
    for q = 1 : length(epsilon)
        for p = 1 : num_exper
            if abs(v(p,1)-mu) > epsilon(q)
                count1 = count1 + 1;
            end
            if abs(v(p,2)-mu) > epsilon(q)
                count2 = count2 + 1;
            end
        end
        p_1(q) = count1/num_exper
        p_2(q) = count2/num_exper;
        count1 = 0;
        count2 = 0;
    end
    plot(epsilon, max(p_1, p_2), epsilon, hoeffding*2)
end