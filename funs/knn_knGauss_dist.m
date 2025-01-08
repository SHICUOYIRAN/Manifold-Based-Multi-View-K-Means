function [knnD] = knn_knGauss_dist(X, Y, s, k)
% k近邻高斯核距离
% Gaussian (RBF) kernel K = exp(-||x-y||/(2s^2));
%   X: d x nx data matrix
%   Y: d x ny data matrix
%   s: sigma of gaussian
%   k: 近邻点
% Ouput:
%   D: k 近邻 kernel distance
arguments 
    X double
    Y double
    s double = 0.5
    k double = 10 % nN/nC, 20
end
[~, nN] = size(X);

EucD = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
EucD = data_process(EucD, "max-min");
K = exp(EucD/(-2 * s^2));
v = diag(K);
D_gauss = v * ones(1,nN) + ones(nN, 1) * v' - 2 * K;
[~, nN] = size(X);
gamma= max(max(D_gauss));
[~, idx] = sort(D_gauss, 2);
knnD = gamma * ones(nN, nN);
id = idx(:, 2:k+1) + (nN.*[0:nN-1]'*ones(1,k));
di = D_gauss(id);
knnD(id) = di + eps;
knnD = knnD - diag(diag(knnD));
knnD = knnD';

end


