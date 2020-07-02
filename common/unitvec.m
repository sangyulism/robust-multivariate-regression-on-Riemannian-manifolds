function uX = unitvec(X)
% UNITVEC converts column vectors in a matrix X into unit column vectors in uX
% 각 X의 열벡터는 크기가 1인 unit vector가 되게 됨.

Z = sqrt(sum(X.^2)); % sum(A)는 각 열의 합을 원소로 가지는 행벡터를 반환
Z(Z < eps) = 1; % eps = 2^(-52), 이는 Z < eps 라면 Z = 1이라는 뜻
uX = X./repmat(Z,size(X,1),1); % repmat 은 Z를 반복해서 만든 size(Z).*[size(X,1),1] 행렬 