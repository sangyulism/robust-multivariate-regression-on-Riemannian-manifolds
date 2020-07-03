function [p, V, E, Y_hat, gnorm] = rmm_sphere_Huber(X, Y, varargin)
%   The result is in p, V, E, Y_hat.
%
%   X is npivots * ndata
%   Y is dimY x ndata column vectors (points on the unit sphere in R^dimY).
%   p is a base point.
%   V is a set of tangent vectors (dimY x dimX).
%   E is the history of the sum of squared geodesic error.
%   Y_hat is the prediction.
%   gnorm is the history of norm of gradients.
    
    ndimY = size(Y,1);
    [ndimX ndata] = size(X);
    Xc = X - repmat(mean(X,2),1,ndata);
    p = karcher_mean_sphere(Y, ones(ndata,1)/ndata, 500);

    logY = logmap_vecs_sphere(p, Y);
    U = null(ones(size(p,1),1)*p');
    Yu = U'*logY;
    L = Yu/Xc;
    V = U*L;
    V = proj_TpM(p,V);

    if nargin >=3
        Huber_delta = varargin{1};
    else
        Huber_delta = 0.1;
    end
    
    if nargin >=4
        maxiter = varargin{2};
    else
        maxiter = 5000;
    end
    
    % Gradient Descent algorithm
    % Step size
    c1 = 1;
    
    % Safeguard parameter
    c2 = 1;

    E = [];
    gnorm = [];
    E = [E; feval_sphere(p,V,X,Y)];
    step = c1;
    for niter=1:maxiter
        Y_hat = prediction_sphere(p,V,X);
        J = logmap_vecs_sphere(Y_hat,Y);
        err_TpM = paralleltranslateAtoB_sphere(Y_hat, p, J);
        err_TpM_Huber = Huber(err_TpM, Huber_delta);
        gradp = -sum(err_TpM_Huber,2);
        gradV = zeros(size(V));
        for iV = 1:size(V,2)
            gradV(:,iV) = -err_TpM_Huber*X(iV,:)';
        end
        gnorm_new = norm([gradV gradp]);
        
        % safeguard
        [gradp gradV] = safeguard(gradp, gradV,c2);
        
        moved = 0;
        for i = 1:200
            step = step*0.5;
            % Safegaurd for gradv, gradp
            p_new = unitvec(expmap_sphere(p,-step*gradp));
            V_new = V -step*gradV;
            V_new = paralleltranslateAtoB_sphere(p,p_new,V_new);
                        
            E_new = feval_sphere(p_new, V_new, X, Y);
            if E(end) > E_new
                p = p_new;
                V = V_new;
                E = [E; E_new];
                gnorm = [gnorm; gnorm_new];
                moved = 1;
                step = min(step*2,1);
                break
            end
            if step < 1e-20
                break
            end
        end
        if moved ~= 1
            break
        end
    end

    E = [E; feval_sphere(p,V,X,Y)];
    Y_hat = prediction_sphere(p,V,X);
end

function V = proj_TpM(p,V)
    for i = 1:size(V,2)
        v = V(:,i);
        V(:,i) = v-p'*v*p;
    end
end 

function [gradp, gradV] = safeguard(gradp, gradV, c2)
    vecnorms = @(A) sqrt(sum(A.^2,1));
    norms = [ vecnorms(gradV) norm(gradp)];
    maxnorm = max(norms);
    if maxnorm > c2
        gradV = gradV*c2/maxnorm;
        gradp = gradp*c2/maxnorm;
    end
    
end

%% Huber
function J_return = Huber(J,Huber_delta)
    J_return = zeros(size(J,1), size(J,2));
    err = zeros(size(J,2),1);
    for i = 1:size(J,2)
        err(i) = norm(J(:,i));
    end
    a = sort(err);
    cutoff = err(round((1-Huber_delta)*size(a,1)));
    for i = 1:size(J,2)
        Ji = J(:,i);
        if err(i) > cutoff
            Ji = Ji/norm(Ji) * cutoff;
        end
        J_return(:,i) = Ji;
    end
end
