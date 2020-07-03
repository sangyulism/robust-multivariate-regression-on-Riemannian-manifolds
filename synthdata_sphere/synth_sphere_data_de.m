% Errors follow a double-exponential distribution

%% Parameters
dimY = 15; % for S^(dimY-1) in R^dimY
npivots = 3;
ndata = 100;

max_noise = 3.0;
noise_size = 0.03;
max_dist_from_p = 3;
mse_iter = 30;

X = rand(npivots,ndata);
X = center(X);

%% Synthesized data for S^(dimY-1)
% npivots = size(X,1); % Number of points except the base point
Yp = zeros(dimY, npivots + 1);
while true
    Yp(:,1) = unitvec(randn(dimY,1));
    for i = 2:(npivots+1)
        Yp(:,i) = addnoise_sphere_normal(Yp(:,1),0.5,2);
    end
    
    V = zeros(dimY,npivots);
    for j = 1:npivots
        V(:,j) = logmap_sphere(Yp(:,1),Yp(:,j+1));
    end
    
    %% Generate Ground Truth
    Y_0 = zeros(dimY, size(X,2));
    issafe = true;
    for i = 1:length(X)
        Vtmp = V*X(:,i);
        if norm(Vtmp) > max_dist_from_p
            issafe = false;
            break
        end
        Y_0(:,i) = expmap_sphere(Yp(:,1),Vtmp);
    end

    if issafe
        break
    end
end

%% Add noise and make some samples.
Ysample = zeros(dimY,size(Y_0,2),mse_iter);

for k = 1:mse_iter
    for i = 1:ndata
        Ysample(:,i,k) = addnoise_sphere_de(Y_0(:,i),noise_size,max_noise);
    end
end

Y_raw = unitvec(Ysample);

function p = randpoint(n)
    x = randn(n,1);
    p = x/sqrt(sum(x .* x));
end
