function P = randspd(n, scale, max_d)
%RANDSPD generates n by b random symmatrix positive matrix P.
%
%   P = randspd(n)
%   P = randspd(n,c)
%   P = randspd(n,c,udit)
%
%   c is parameter for variance. Bigger c has bigger variance.
%   udist is the upper bound of distance from I to P w.r.t. GL-invariant
%   measure.
%
%   See also SYNTH_DTI_DATA

%   Hyunwoo J. Kim
%   $Revision: 0.1 $  $Date: 2014/06/23 16:03:38 $
% if nargin >= 2
%     c = varargin{1};
% else
%     c = 3;
% end

    P = scale*(rand(n)-0.5);
    P = P*P';
    while dist_M_spd(P,eye(n)) > max_d %eye(n) 은 In
        P = scale*(rand(n)-0.5); % rand(n)은 unif[0,1]로 구성된 n*n matrix
        P = P*P';
    end
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