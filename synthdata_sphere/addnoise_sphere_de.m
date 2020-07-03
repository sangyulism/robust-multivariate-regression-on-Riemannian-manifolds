% Sampling Double-exponential error on S^(n-1) in R^n
function Anew = addnoise_sphere_de(A, err_scale, maxerr)
    n = size(A,1);
    V = randn(size(A));
    V = V-V'*A*A; % Project V on T_A(M)
    V = sphere_rand_de(n,err_scale,maxerr) * V / norm(V);
    Anew = expmap_sphere(A, V);
end

% f(k) ~ exp(-k/c) * sin(k)^(n-2) (0 < k < maxerr < pi) and 0 (o.w.)
function x = sphere_rand_de(n,c,maxerr)
    if maxerr < (n-2) * c
        M = exp(-maxerr/c) * sin(maxerr)^(n-2);
    else
        M = exp(-(n-2)) * (n-2)^(n-2) * c^(n-2);
    end
    while true
        k = rand * maxerr;
        if rand < exp(-k/c) * sin(k)^(n-2) / M
            x = k;
            break
        end
    end
end