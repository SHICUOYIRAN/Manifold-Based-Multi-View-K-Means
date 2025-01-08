function [alpha] = auto_weight(M, r)
%  min_alpha \sum_v {alpha_v^r * M_v}
%  s.t.  \sum_v{alpha_v} = 1,  alpha_v >= 0
nV = length(M);
Sum = 0;
if (r == 1)
    error("权重的指数不能为1")
end
for v = 1:nV
    M(v) = M(v) ^ (1 / (1 - r));
    Sum = Sum + M(v);
end
alpha = M / Sum;

end

