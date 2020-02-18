numCoin = 6;
mu = 0.5;
pdf = [];
for i = 0:6
    pdf = [pdf, binopdf(i,numCoin,mu)];
end
pdf
epsilon = 0:0.001:1;
Pcoin1 = [];
Pcoin2 = [];
epsilonNum = length(epsilon);
v = (0:6)/6;
for m = 1 : epsilonNum
    p1 = 0;
    p2 = 0;
    for i = 1:length(v)
        if abs(v(i)-mu) > epsilon(m)
            p1 = p1 + pdf(i);
            p2 = p2 + pdf(i);
        end
    end
    Pcoin1 = [Pcoin1, p1];
    Pcoin2 = [Pcoin2, p2];
end
P = max(Pcoin1, Pcoin2);
Hbound = 2*2*exp(-2*numCoin*epsilon.^2);

plot(epsilon, P, epsilon, Hbound)
title('Max probability between two coins with Hoeffding bound')
xlabel('epsilon')
ylabel('Probability')
legend('Max Prob. of diff. betwn. v_i and \mu greater than \epsilon','Hoeffding', 'Location','Northeast')