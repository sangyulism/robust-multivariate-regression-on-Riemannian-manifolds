function [p, V, E, Y_hat, gnorm, Numerical_error] = mglm_spd_huber(X, Y, varargin)
%MGLM_SPD performs MGLM on SPD manifolds by interative method.
%
%   [p, V, E, Y_hat, gnorm] = MGLM_SPD(X, Y)
%   [p, V, E, Y_hat, gnorm] = MGLM_SPD(X, Y, MAXITER)
%   has optional parameter MAXITER.  
%
%   The result is in p, V, E, Y_hat.
%
%   X is dimX x N column vectors
%   Y is a stack of SPD matrices. 3D arrary 3x3xN.
%   p is a base point.(spd(3)의 원소, 즉 3*3*1 matrix)
%   V is a set of tangent vectors (3 x 3 x dimX symmetric matrix).
%   E is the history of the sum of squared geodesic error.
%   Y_hat is the prediction.
%   gnorm is the history of norm of gradients.
%
%   See also WEIGHTEDSUM_MX, MGLM_LOGEUC_SPD

%   Hyunwoo J. Kim
%   $Revision: 0.1 $  $Date: 2014/06/23 00:13:20 $
    Numerical_error = 0;
    ndimX = size(X,1);
    ndimY = size(Y,1);
    ndata =  size(X,2);
    
    if ndata ~= size(Y,3)
        error('Different number of covariate X and response Y')
    end

    % Initialization
    p = karcher_mean_spd(Y,[],500);
    % V = zeros([ndimY ndimY ndimX]);
    logY = logmap_vecs_spd(p, Y);
    Xc = X - repmat(mean(X,2),1,ndata);
    Yv = embeddingR6_vecs(p,logY);
    L = Yv/Xc;
    logYv_hat = L*Xc;
    V_hat = invembeddingR6_vecs(p,logYv_hat); 
    V = invembeddingR6_vecs(p,L);
    V = proj_TpM_spd(V);

    if nargin >=3
        huber_delta = varargin{1};
    else
        huber_delta = 1.345;
    end
    
    if nargin >=4
        maxiter = varargin{2};
    else
        maxiter = 5000;
    end

    % Gradient Descent algorith
    % Step size
    c1 = 1;
    
    % Safeguard parameter
    c2 = 1;

    E = [];
    gnorm = [];
    E = [E; feval_spd(p,V,X,Y)]; %feval_spd은 error
    step = c1;
    for niter=1:maxiter
        Y_hat = prediction_spd(p,V,X);
        J = logmap_vecs_spd(Y_hat, Y);
        err_TpM = paralleltranslateAtoB_spd(Y_hat, p, J);
        err_TpM_huber = huber(err_TpM, huber_delta, p);
        gradp = -sum(err_TpM_huber,3); % 3th axis를 따라 더함. 3*3*1 matrix가 나오게 됨
        
        % v projection on to tanget space
        gradV = zeros(size(V));
        
        % Matrix multiplicaton
        for iV = 1:size(V,3)
            gradV(:,:,iV) = -weightedsum_mx(err_TpM_huber,X(iV,:));
        end
        
        ns = normVs(p,gradV);
        normgradv = sum(ns);
        
        ns = normVs(p,gradp);
        normgradp = sum(ns);

        gnorm_new = normgradp+normgradv;
        if ~isreal(gnorm_new)
            disp('Numerical Error huber.1');
            Numerical_error = 1;
            %exit % 이거 원래 없는 건데, disp('Numerical Error.2')밑에서 자꾸 에러가 떠서 걍 미리 나가게..
        end

        if Numerical_error == 1
            break
        end

        % Safegaurd
        [gradp gradV] = safeguard(gradp, gradV, p, c2);
        
        % 두 번째 for 문에서는 step을 1/2씩 줄여나가면서(최대 50번) Error를 줄이는 방향 찾기. 실패하면 끝냄.
        moved = 0;
        for i = 1:50
            step = step*0.5;
            % Safegaurd for gradv, gradp
            V_new = V -step*gradV;
            p_new = expmap_spd(p,-step*gradp);
            if ~isspd(p_new)
                p_new = proj_M_spd(p_new);
            end
            V_new = paralleltranslateAtoB_spd(p,p_new,V_new);
            E_new = feval_spd(p_new, V_new, X, Y);
            
            if E(end) > E_new
                p = p_new;
                V = proj_TpM_spd(V_new);
                E = [E; E_new];
                
                if ~isreal(gnorm_new)
                    disp('Numerical Error huber.2');
                    disp(p);
                    disp(V_new);
                    Numerical_error = 1;
                    break
                end
                
                gnorm = [gnorm; gnorm_new];
                moved = 1;
                step = step*2;
                break
            end
        end
        if Numerical_error == 1
            break
        end
        if moved ~= 1 || gnorm(end) < 1e-10 % 50번해도 moved =1 안되거나, gnorm(gradiant의 norm의 합)이 충분히 작으면 끝.
            break
        end
    end

    if Numerical_error == 1
        E = 0;
        Y_hat = 0;
    end

    E = [E; feval_spd(p,V,X,Y)];
    Y_hat = prediction_spd(p,V,X);
end

%% NormVs
% size(V,3) * 1 벡터
function ns = normVs(p,V)
    for i =1:size(V,3)
        ns(i,1) = norm_TpM_spd(p,V(:,:,i));
    end
end
 
%% Safeguard
% gradp,gradV 각 층의 norm의 합이 c2를 넘지않도록 함
function [gradp gradV] = safeguard(gradp, gradV, p, c2)
    ns = normVs(p,gradV);
    normgradv = sum(ns);
    ns = normVs(p,gradp);
    normgradp = sum(ns);
    norms = [ normgradp normgradv];
    maxnorm = max(norms);
    if maxnorm > c2
        gradV = gradV*c2/maxnorm;
        gradp = gradp*c2/maxnorm;
    end
end

%%huber
% 3*3*ndata matric J 에 대해 (즉, J는 ndata개의 3*3 sym mat 인 vector 들) huber form을 계산
% 여기서 계산은 단순히 vector의 norm이 huber_delta 이상이면, 상수로 나옴.

% quantile(0.1788) 야매
function J_return = huber(J,huber_delta,p)
    J_return = zeros(size(J));
    err = zeros(size(J,3),1);
    for i = 1:size(J,3)
        err(i) = norm_TpM_spd(p, J(:,:,i));
    end
    a = sort(err);
    cutoff = err(round(0.8212*size(a,1)));
    for i = 1:size(J,3)
        Ji = J(:,:,i);
        if err(i) > cutoff
            Ji = Ji/err(i) * cutoff;
        end
        J_return(:,:,i) = Ji;
    end
end

% 좀 더 정확하게 scale 추정하는 코드 만들어야함.

%이거 잘못된건데 왤캐 퍼포먼스가 좋지;
% function J_return = huber(J,huber_delta,p)
%     J_return = zeros(size(J));
%     err = zeros(size(J,3),1);
%     for i = 1:size(J,3)
%         err(i) = norm_TpM_spd(p, J(:,:,i));
%     end
%     s = median(abs(err - median(err)))/0.6745;
%     for i = 1:size(J,3)
%         Ji = J(:,:,i);
%         if err(i) > huber_delta * s
%             Ji = Ji/err(i) * huber_delta * s;
%         end
%         J_return(:,:,i) = Ji;
%     end
% end