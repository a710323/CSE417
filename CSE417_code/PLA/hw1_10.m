numCoin = 6;
mu = 0.5;
pdf = [];
for i = 0:6
    pdf = [pdf, binopdf(i,numCoin,mu)];
end
pdf
epsl = 0:0.001:1;
Pcoin1 = [];
Pcoin2 = [];
epslNum = length(epsl);
v = (0:6)/6;
for m = 1:epslNum
    p1 = 0;
    p2 = 0;
    for i = 1:length(v)
        if abs(v(i)-mu) > epsl(m)
            p1 = p1 + pdf(i);
            p2 = p2 + pdf(i);
        end
    end
    Pcoin1 = [Pcoin1, p1];
    Pcoin2 = [Pcoin2, p2];
end
P = max(Pcoin1, Pcoin2);
Hbound = 2*2*exp(-2*numCoin*epsl.^2);
plot(epsl, P, epsl, Hbound, 'r')