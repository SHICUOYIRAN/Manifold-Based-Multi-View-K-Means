
function [Out] = data_process(In, method)

    arguments
        In
        method (1,:) char {mustBeMember(method,{'maxAbs','max-min','norm','std'})} = 'std'
    end
    if iscell(In)
        nV = length(In);
        X = In;
    else 
        nV = 1;
        X{1} = In;
    end
    % disp(['Preprocessing: ' method ';']);
    if method == "maxAbs"
        for v=1:nV
            a = max(X{v}(:));
            X{v} = double(X{v}./a);
        end
    elseif method == "max-min"
        for v=1:nV
            X{v} = (X{v} - min(X{v}(:))) / (max(X{v}(:)) - min(X{v}(:)));
        end
    elseif method == "norm"
        for v=1:nV
             X{v} = X{v}./sqrt(diag(X{v}*X{v}'));
        end
    elseif method == "std"
        for v=1:nV
            X{v} = zscore(X{v});
        end
    end
    if iscell(In) 
        Out = X;
    else 
        Out = X{1};
    end
end

