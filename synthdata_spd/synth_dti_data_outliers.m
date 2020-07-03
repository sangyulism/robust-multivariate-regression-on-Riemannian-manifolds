% Errors follow a Gaussian distribution, but 10% of the data is assumed to be anomalous.

%% Parameters
ndata = 100;
npivots = 3 ;
noise_size = 0.1;
max_noise = 1;
mse_iter = 30;
noutliers = 10;

%% Synthesize Ground Truth
X = rand(npivots,ndata);
X = center(X);

%% Synthesized data for PD(3)
% npivots = size(X,1); % Number of points except the base point
Yp = zeros(3,3,npivots+1);

Yp(:,:,1) = randspd(3,2,3);
for i = 2:(npivots+1)
    Yp(:,:,i) = randspd(3,2,10);
end

% Tangent vectors, geodesic bases.
V = zeros(3,3,npivots);
for j =1:npivots
    V(:,:,j) = logmap_spd(Yp(:,:,1),Yp(:,:,j+1));
end

%% Generate Ground Truth Data
Y0 = zeros(3,3,ndata);
for i = 1:ndata
    Vtmp = zeros(3,3,1);
    for j =1:npivots
        Vtmp = Vtmp+ V(:,:,j)*X(j,i);
    end
    Y0(:,:,i) = expmap_spd(Yp(:,:,1),Vtmp);
end

%% Sanity check
notspd = 0;
for i=1:ndata
    notspd = notspd + (~isspd(Y0(:,:,i)));
end

assert(notspd ==0)
%%

Ysample = zeros(3,3,ndata,mse_iter);
for k = 1:mse_iter
    for i = 1:ndata
        Ysample(:,:,i,k) = addnoise_spd_normal(Y0(:,:,i),noise_size,max_noise);
    end
    for i = (ndata - noutliers+1):ndata
        % Ysample(:,:,i,k) = randspd(3,2,0.8);
        % Ysample(:,:,i,k) = Ysample(:,:,i,k) * 2;
        Ysample(:,:,i,k) = inv(Ysample(:,:,i,k))*2;
    end
end

assert(isspd_mxstack(Ysample) == 1)

Y_raw = Ysample;

