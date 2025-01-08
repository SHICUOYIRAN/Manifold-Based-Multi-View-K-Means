function D = knGauss_dist(X, Y, s)
% Gaussian (RBF) kernel K = exp(-|x-y|/(2s));
% Input:
%   X: d x nx data matrix
%   Y: d x ny data matrix
%   s: sigma of gaussian
% Ouput:
%   K: nx x ny kernel matrix
%   D: kernel distance
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    s = 1;
end
[~,nN] = size(X);
if nargin < 2 || isempty(Y)  
    K = ones(1,size(X,2));            % norm in kernel space
else
    EucD = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
    EucD = data_process(EucD, 'max-min');
    K = exp(EucD/(-2*s^2));
    v = diag(K);
    D = v * ones(1,nN) + ones(nN, 1) * v' - 2 * K;
end

