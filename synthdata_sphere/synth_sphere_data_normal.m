% 이 코드는 수정되어서 같은 X를 여러개 붙인게 아니라 그냥 npair개 만큼 생성할껍니다.

%% Parameters
dimY = 15;
npivots = 3;
npairs = 100;

max_noise_size = 3;
noise_size = 0.03;
udist = 3; % udist : V의 크기를 결정하는 parameter
noutliers = 0;
mse_iter = 30;

%% For figure
X = rand(npivots,npairs);
X = center(X);

%% Synthesized data
%npivots = size(X,1); % Number of points except the base point
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
        if norm(Vtmp) > udist
            issafe = false;
            break
        end
        Y_0(:,i) = expmap_sphere(Yp(:,1),Vtmp);
    end

    if issafe
        break
    end
end

%% outlier
Yp_outlier = zeros(dimY, npivots + 1);
while true
    Yp_outlier(:,1) = unitvec(randn(dimY,1));
    for i = 2:(npivots+1)
        Yp_outlier(:,i) = addnoise_sphere_normal(Yp_outlier(:,1),0.5,2);
    end
    V_outlier = zeros(dimY,npivots);
    for j = 1:npivots
        V_outlier(:,j) = logmap_sphere(Yp_outlier(:,1),Yp_outlier(:,j+1));
    end
    
    %% Generate Ground Truth
    Y_0_outlier = zeros(dimY, size(X,2));
    issafe = true;
    for i = 1:length(X)
        Vtmp_outlier = V_outlier*X(:,i);
        if norm(Vtmp_outlier) > udist
            issafe = false;
            break
        end
        Y_0_outlier(:,i) = expmap_sphere(Yp_outlier(:,1),Vtmp_outlier);
    end

    if issafe
        break
    end
end

%% Add noise and make some samples.
Ysample = zeros(dimY,size(Y_0,2),mse_iter);

for k = 1:mse_iter
    for i = 1:(npairs - noutliers)
        Ysample(:,i,k) = addnoise_sphere_normal(Y_0(:,i),noise_size,max_noise_size);
    end
    for i = (npairs - noutliers+1):npairs
        Ysample(:,i,k) = addnoise_sphere_normal(Y_0_outlier(:,i),noise_size,max_noise_size);
    end
end

Y_raw = unitvec(Ysample);
