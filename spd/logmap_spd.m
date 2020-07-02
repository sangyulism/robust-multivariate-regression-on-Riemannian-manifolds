function v = logmap_spd(P,X)
%LOGMAP_SPD maps X on SPD manifold to the tangent space at P.
%
%    v = LOGMAP_SPD(P,X)
%
%    P, X is a SPD matrix.
%    v is a symmetric matrix.
%
%   See also EXPMAP_SPD, INNERPROD_TPM_SPD, DIST_M_SPD

%   Hyunwoo J. Kim
%   $Revision: 0.1 $  $Date: 2014/06/23 15:20:53 $

if norm(P-X) < 1e-18
    v = zeros(size(P));
    return
end

[U D] = eig(P); % U 는 각 열벡터가 P의 eigen vector, D는 각 값이 eigen value 인 diagonal matrix, P가 spd 이므로 P = UDU' (UU'=U'U=I),D의 각 원소는 양수
g = U*sqrt(D); % P=gg'
invg = inv(g);
y = invg*X*invg';
[V S] = eig(y);
H = g*V;
v = H*diag(log(diag(S)))*H';

%rtX = sqrtm(X);
%invrtX = inv(rtX);
%v = rtX*logm(invrtX*Y*invrtX)*rtX;
% v = (v+v')/2;