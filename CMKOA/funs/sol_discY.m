
function [Y] = sol_discY(D, beta, HH, mu, HW, Y_pre)
% Solving the following model:
%    min_Y tr(Y'DY) -beta * ||Y||_* + mu/2 * ||Y - H||_F^2
%    s.t. Y \in Ind, D_ii = 0
%

[nN, ~] = size(HW);
Y = Y_pre;
for i = 1:nN 
    M = (2*(D(i,:)*Y) -beta *HH(i,:) - mu * HW(i,:))';%*lambda
    [~, m] = min(M);
    Y(i,:) = 0;
    Y(i,m) = 1;
end

end


