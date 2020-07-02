% 점 p 에 대해 err가 isotropic하게 새로운 점 q를 반환
% n =3
function q = addnoise_spd_de(p, err_scale, maxerr)
    D = de_D(err_scale,maxerr);
    U = randortho(3);
    rtp = sqrtm(p);
    q = rtp * (U * D * U') * rtp';
end

% S = randS(size(p,1));
% rtp = sqrtm(p);
% q = rtp * expm(err_scale * S) * rtp;

% n 차원 unit sphere 에서 uniform random하게 한 점 뽑기
function p = randpoint(n)
    x = randn(n,1);
    p = x/sqrt(sum(x .* x));
end

% uniform random 하게 n * n orthonormal matrix 뽑기
function ortho = randortho(n)
    X = zeros(n);
    for i = 1 : n
        X(:,i) = randpoint(n);
    end
    ortho = gram(X);
end

% uniform random 하게 tr(S^2) = 1 인 S 뽑기
function S = randS(n)
    x = randpoint(n);
    D = diag(x);
    U = randortho(n);
    S = U * D * U';
end

% 그람-슈미트 방법. 인수는 n * n 행렬, 반환인자도 n * n 행렬이고, 각 열 벡터에 대해 그람슈미트 방법을 적용.
function Y = gram(X)
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

% function D = de_D(err_scale,maxerr)
%     while true
%         while true
%             r = randn(1,3)*err_scale;
%             if norm(r) < maxerr
%                 break
%             end
%         end
%         if rand * exp((norm(maxerr)^2-norm(maxerr))/(2*err_scale^2)) < exp((norm(r)^2-norm(r))/(2*err_scale^2)) ...
%             && rand < exp(-2*maxerr) * sinh(abs(r(1)-r(2))/2) * sinh(abs(r(1)-r(3))/2) * sinh(abs(r(2)-r(3))/2)
%             break
%         end
%     end
%     D = expm(diag(r));
% end