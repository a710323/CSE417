function [ z ] = computeLegPoly_hu( x, Q )
%COMPUTELEGPOLY Return the Qth order Legendre polynomial of x
%   Inputs:
%       x: vector (or scalar) of reals in [-1, 1]
%       Q: order of the Legendre polynomial to compute
%   Output:
%       z: matrix where each column is the Legendre polynomials of order 0 
%          to Q, evaluated atthe corresponding x value in the input

L0 = 1;
z_x = ones(Q+1,1);
z_x(1) = L0;
z = ones(Q+1,length(x));
for i = 1:length(x)
    L1 = x(i);
    z_x(2) = L1;
    for j = 3:Q+1
      L = (2*(j-1)-1)/(j-1)*x(i)*z_x(j-1)-((j-1)-1)/(j-1)*z_x(j-2);
      z_x(j) = L;
    end
    z(:,i) = z_x;
end
       
end