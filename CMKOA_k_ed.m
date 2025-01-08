clear
close all
addpath('.\datasets');
addpath('.\funs');

dataname = 'ORL'; lambda=100; r=3; p=0.1; sigma = 0.2;beta=[0.0015];%1 1 1
load([dataname '.mat']);
gt=double(gt);
nV = size(X, 2);
[dv, nN] = size(X{1}'); %[样本维数，总样本数目]
nC = length(unique(gt)); 
%% ================ Parameter Setting =========================== 
Iter_max = 200;
max_mu = 12e12; 
rho = 1.1;


%% ================ Input Distance ================================

for sigma = sigma
D = cell(1,nV);
for v = 1:nV
    X{v} = data_process(X{v}, "max-min");
    D{v} = knGauss_dist(X{v}', X{v}', sigma);
end


for p = p
for r = r
for lambda = lambda
for beta=beta
%% =============================== main =================================
tic
% initialize Y J Q H
Y = cell(1, nV);
J = cell(1, nV);
Q = cell(1, nV);
for i = 1:nN
    Y{1}(i,mod(i, nC)+1) = 1;
end

for v = 1:nV
    Y{v} = Y{1};
    J{v} = zeros(nN, nC);
    Q{v} = zeros(nN, nC);
end
alpha = ones(1,nV)./nV;
mu = 0.0001; 

% ---------- Iterative Update -----------
obj = [1];
obj_Y = [];
flag = 1;
iter = 1;
while flag == 1  
    % Solving HH
    for v = 1:nV
        [a1,a2,a3]=mySVD(full(Y{v})); 
        HH{v} =  a1*pinv(full(a2))*abs(a2)*a3';   
    end
    % Solving Y
    for v = 1:nV
        HW =  J{v} - Q{v}/mu;
        Y{v} = sol_discY(D{v},beta / alpha(v)^r, HH{v}, mu / alpha(v)^r, HW, Y{v});
    end
    % Solving J --Schatten p_norm
    [J, Q, mu] = sol_TSP(lambda, mu, p, Y, Q, rho, max_mu);
    
    % Solving alpha 
    for v = 1:nV
        h(v) = trace(Y{v}'*D{v}*Y{v});
    end
    alpha = auto_weight(h, r);

    oo = 0;
    for v = 1:nV
        [a1,a2,a3]=mySVD(full(Y{v}));
        oo = oo + norm(Y{v} - J{v},'fro');
    end

    obj_Y = [obj_Y oo];
    % ShouLian
    if  obj_Y(iter) < 1e-8 || iter == Iter_max
        flag = 0;
    end
    iter = iter + 1;
end
% plot(obj_Y(2:iter-1))
time = toc;
%% ================== Perfermance Calculate=======================
Y_sum = zeros(nN, nC);

for v=1:nV
    Y_sum = Y_sum + alpha(v)^r*Y{v};                                  
end
[~, label] = max(Y_sum');
result = ClusteringMeasure(gt,label);
fprintf('result=%.4f,%.4f,%.4f | lambda=%.4f,beta=%.5f,r=%.1f,p=%.1f,time(s)=%.2f,iter=%d\n',result(1:3),lambda,beta,r,p,time,iter);


end
end
end
end
end