function [dist] = mydistance(A, B, method, k)
%
% calculate the distance between A and B;
% default k = 10
%
arguments
    A double 
    B double
    method  (1,:) char {mustBeMember(method,{'L1','L2','knn-L2'})} = 'L2'
    k = 10
end

switch(method)
    case 'L1'
        dist = L1_distance(A, B);
    case 'L2'
        dist = L2_distance_1(A', B');
    case 'knn-L2'
        dist = kNN_L2_dist(A', k);
end

end

%% L1
function D = L1_distance(A,B)
    nN = size(A,1);
    nM = size(B,1);
    D = zeros(nN,nM);
    for i = 1:nN
        D(i,:) = sum(abs(A(i,:)-B),2);
    end
end

%% knn_L2
% function [D] = kNN_L2_distance(A, B, k)
% arguments
%     A double
%     B double
%     k = 10;
% end
% [nN, ~] = size(A);
% D_temp = L2_distance_1(A', B');
% gamma= max(max(D_temp));
% [distXs, idx] = sort(D_temp, 2);
% D = gamma * ones(nN,nN);
% id = idx(:, 2:k+1) + (nN.*[0:nN-1]'*ones(1,k));
% di = D_temp(id);
% D(id) = di + eps;
% D = D - diag(diag(D));
% D = D';
% end
function [D] = kNN_L2_dist(X, k)
arguments
    X double
    k = 10;
end
[~, nN] = size(X);
D_temp = L2_distance_1(X,X);
gamma= max(max(D_temp));
[~, idx] = sort(D_temp, 2);
D = gamma * ones(nN,nN);
id = idx(:, 2:k+1) + (nN.*[0:nN-1]'*ones(1,k));
di = D_temp(id);
D(id) = di + eps;
D = D - diag(diag(D));
D = D';
end

function d = L2_distance_1(a,b)
% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b
if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end
aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;
d = real(d);
d = max(d,0);
end
