% Sampling Double-exponential error on PD(3).
function q = addnoise_spd_de(p, err_scale, maxerr)
    D = de_D(err_scale,maxerr);
    U = randortho(3);
    rtp = sqrtm(p);
    q = rtp * (U * D * U') * rtp';
end

% Sampling p uniformly distributed on S^n
function p = randpoint(n)
    x = randn(n,1);
    p = x/sqrt(sum(x .* x));
end

% Sampling U uniformly distributed on O(n)
function U = randortho(n)
    X = zeros(n);
    for i = 1 : n
        X(:,i) = randpoint(n);
    end
    U = GramSchmidt(X);
end

% Gram-Schmidt
function Y = GramSchmidt(X)
    Y = zeros(size(X));
    for k = 1 : size(X,1)
        proj_temp = zeros(size(X,1),1);
        for j = 1:(k-1)
            proj_temp = proj_temp + proj_uv(Y(:,j),X(:,k));
        end
        Y(:,k) = X(:,k) - proj_temp;
    end
    for k = 1 : size(X,1)
        Y(:,k) = Y(:,k)/norm(Y(:,k));
    end
end

function proj = proj_uv(u,v)
    proj = sum(u .* v)/sum(u .* u) * u;
end

function D = de_D(err_scale,maxerr)
    while true
        while true
            k = abs(log(rand/rand))*err_scale;
            if k < maxerr && rand * maxerr^2 < k^2
                break
            end
        end
        r = randn(1,3);
        r = r/norm(r)*k;
        if rand < exp(-2*maxerr) * sinh(abs(r(1)-r(2))/2) * sinh(abs(r(1)-r(3))/2) * sinh(abs(r(2)-r(3))/2)
            break
        end
    end
    D = expm(diag(r));
end