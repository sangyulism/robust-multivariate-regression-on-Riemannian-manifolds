function p = proj_M_spd(X,varargin)
%PROJ_M_SPD projects a matrix onto SPD manifolds.
%
%
%    Example:
%        p = PROJ_M_SPD(X)s
%
%   p is the point on SPD manifolds.
%   X is a n x n matrix.
%
%   See also MGLM_LOGEUC_SPD, MGLM_SPD

%   Hyunwoo J. Kim
%   $Revision: 0.1 $  $Date: 2014/06/23 16:40:17$ 

    if nargin == 2
        c = varargin{1};
    else 
        c = eps; % eps = 2^(-52)
    end
    
    % Make a matrix symmetric positive definite.
    if norm(X-X') > eps %norm(X)는 X의 largest singular value
        X = (X+X')/2;
    end
    [V D ] = eig(X);
    D = diag(D);
    p = zeros(size(X));
    for i =1:length(D)
        if D(i) > 0
            p = p + D(i)*V(:,i)*V(:,i)';
        end
    end
    % Now X is spd
    % Make psd matrix (psd = positive semidefinite)
    if sum(D > 0+c) < length(D) %D의 원소중 eps 보다 작은게 있다면, 
        a = 1e-16; 
        pnew = p;
        while ~isspd(pnew, c)
            pnew = p + a*eye(3);
            a = 2*a;
        end
        p = pnew;
    end
end