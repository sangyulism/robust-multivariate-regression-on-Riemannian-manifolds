% Sampling Gaussian error on S^(n-1) in R^n
function Anew = addnoise_sphere(A, err_scale, maxerr)
    n = size(A,1);
    V = randn(size(A)); 
    V = V-V'*A*A; % Project V on T_A(M)
    V = addnoise_sphere_sub(n,err_scale,maxerr) * V / norm(V);
    Anew = expmap_sphere(A, V);
end

% f(k) ~ exp(-k^2/2c^2) * sin(k)^(n-2) (0 < k < maxerr < pi) and 0 (o.w.)
function x = addnoise_sphere_sub(n,c,maxerr)
    M = exp(-(n-2)/2) * (n-2)^((n-2)/2) * c^(n-2);
    while true
        k = rand * maxerr;
        if rand < exp(-k^2/(2*c^2)) * sin(k)^(n-2) / M
            x = k;
            break
        end
    end
end
