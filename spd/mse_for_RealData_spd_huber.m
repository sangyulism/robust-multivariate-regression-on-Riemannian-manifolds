function [mse_return] = mse_for_RealData_spd_huber(X, Y, varargin)
% real data 에서 mse 계산해줌
% X : npivots * ndata, Y : 3 * 3 * ndata

    ndata = size(Y,3);
    if ndata ~= size(X,2)
        error('Different number of covariate X and response Y')
    end
    npivots = size(X,1);
    ndimY = size(Y,1);

    mse_return = 0;
    for i  = 1:ndata
        Xtmp = X(:,[1:(i-1) (i+1):ndata]);
        Ytmp = Y(:,:,[1:(i-1) (i+1):ndata]);
        Yi = squeeze(Y(:,:,i,:));
        [p, V, E, Y_hat, gnorm] = mglm_spd_huber(Xtmp,Ytmp);
        Vtmp = zeros(ndimY,ndimY,1);
        for j =1:npivots
            Vtmp = Vtmp+ V(:,:,j)*X(j,i);
        end
        Yi_hat = expmap_spd(p,Vtmp);
        err = logmap_spd(Yi,Yi_hat);
        err_transpose = paralleltranslateAtoB_spd(Yi, eye(ndimY),err);
        mse_return = mse_return + norm_TpM_spd(eye(ndimY),err_transpose)^2;
    end
    mse_return = mse_return/ndata;
end

