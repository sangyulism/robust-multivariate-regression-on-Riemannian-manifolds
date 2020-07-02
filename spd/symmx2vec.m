function v = symmx2vec(mx)
%SYMMX2VEC converts matrices mx to vectors v. 
%
%   v = symmx2vec(mx)
%
%   v is a set of n(n+1)/2 dimensional vectors to n by n matrices.
%   mx is a set of n x n matrices.
%
%   See also INVEMBEDDINGR6, VEC2SYMMX

%   Hyunwoo J. Kim
%   $Revision: 0.1 $  $Date: 2014/06/23 15:09:53 $

% v 의 각 행은 mx의 upper 부분을 1행은 1~n열, 2행은 2~n열,..., n행은 n열 해서 나온 벡터들 
% n * n * ndata matrix mx를 (n(n+1)/2) * ndata 행렬 v 로 반환.
    [ nrow ncol ndata ] = size(mx);
    v = zeros(nrow*(nrow+1)/2,ndata);
    k =1;
    for i=1:ncol
        for j=i:ncol
            v(k,:) = squeeze(mx(i,j,:))';
            k = k + 1;
        end

    end
end