function [J, Q, mu] = sol_TSP(lambda, mu, p, Y, Q, rho, max_mu)
% J = argmin_J lambda ||J||_Sp^p + mu / 2 ||Y - J + Q / mu||_F^2
% J, Y, Q are Tensor

nV = length(Y);
% solve J
for v =1:nV
    QQ1{v} = Y{v} + Q{v} ./ mu;
end
Q_tensor = cat(3, QQ1{:, :});
Qg = Q_tensor(:);
[nN, nC] = size(Y{1});
sX = [nN, nC, nV];
[myj, ~] = wshrinkObj_weight_lp(Qg, ones(1, nV)'.*(1 * lambda/mu), sX, 0, 3, p);
J_tensor = reshape(myj, sX);
for v=1:nV
    J{v} = J_tensor(:,:,v);
    Q{v} = Q{v} + mu*(Y{v}-J{v});
end
mu = min(rho*mu, max_mu);
end

