% size(X,2) 개 만큼의 (x,y)를 생성함. x는 npivots 차원, y는 spd(3)
% X는 x가 size(X,2)개, Xsample 은 이런 X를 npairs 개 옆으로 붙여놓음
% Y는 X의 각 열벡터 x에 대응하는 열벡터들로 구성되어 있으며, Ysample은 noise를 추가하여 npairs개 붙여놓음
% 결론적으로 밑의 코드에서 Xsample은 2*100, Ysample은 3*3*100

%% Parameters
npairs = 100;
npivots = 3 ;
noise_size = 0.05;
max_noise = 1;
mse_iter = 30;
noutliers = 0;

%% Synthesize Ground Truth
% X1 = [0:0.25:1 0:0.25:1];    
% X2 = [0 0 0 0 0 1 1 1 1 1 ];
% X = [X1;X2];
X = rand(npivots,npairs);
X = center(X);

% npivots = size(X,1); % Number of points except the base point
Yp = zeros(3,3,npivots+1);

%p 생성: Yp(:,:,1), 그리고 Yp(,,2)-Yp(,,1),Yp(,,3)-Yp(,,1)은 각각 tangent vector
% Yp(:,:,1) = eye(3); % temp test code
Yp(:,:,1) = randspd(3,2,3); % 3*3 random spd 생성, 2는 variance parameter, 3은 In으로부터 거리가 3이하가 되라는 뜻
for i = 2:(npivots+1)
    Yp(:,:,i) = randspd(3,2,10);
end

% Tangent vectors, geodesic bases.
V = zeros(3,3,npivots);
for j =1:npivots
    V(:,:,j) = logmap_spd(Yp(:,:,1),Yp(:,:,j+1));
end

%% Generate Ground Truth Data
% Y(:,:,i) = p + VX(:,i) 생성
Y0 = zeros(3,3,npairs);
for i = 1:npairs
    Vtmp = zeros(3,3,1);
    for j =1:npivots
        Vtmp = Vtmp+ V(:,:,j)*X(j,i);
    end
    Y0(:,:,i) = expmap_spd(Yp(:,:,1),Vtmp);
end

%% Sanity check
notspd = 0;
for i=1:npairs
    notspd = notspd + (~isspd(Y0(:,:,i)));
end

assert(notspd ==0)
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%% outlier

% npivots = size(X,1); % Number of points except the base point
Yp_outlier = zeros(3,3,npivots+1);

%p 생성: Yp(:,:,1), 그리고 Yp(,,2)-Yp(,,1),Yp(,,3)-Yp(,,1)은 각각 tangent vector
Yp_outlier(:,:,1) = randspd(3,2,3); % 3*3 random spd 생성, 2는 variance parameter, 3은 In으로부터 거리가 3이하가 되라는 뜻
for i = 2:(npivots+1)
    Yp_outlier(:,:,i) = randspd(3,2,10);
end

% Tangent vectors, geodesic bases.
V_outlier = zeros(3,3,npivots);
for j =1:npivots
    V_outlier(:,:,j) = logmap_spd(Yp_outlier(:,:,1),Yp_outlier(:,:,j+1));
end

%% Generate Ground Truth Data
% Y(:,:,i) = p + VX(:,i) 생성
Y0_outlier = zeros(3,3,npairs);
for i = 1:npairs
    Vtmp_outlier = zeros(3,3,1);
    for j =1:npivots
        Vtmp_outlier = Vtmp_outlier+ V_outlier(:,:,j)*X(j,i);
    end
    Y0_outlier(:,:,i) = expmap_spd(Yp_outlier(:,:,1),Vtmp_outlier);
end

%% Sanity check
notspd = 0 ;
for i=1:npairs
    notspd = notspd + (~isspd(Y0_outlier(:,:,i)));
end
assert(notspd ==0)
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ysample = zeros(3,3,npairs,mse_iter);
for k = 1:mse_iter
    for i = 1:(npairs - noutliers)
        Ysample(:,:,i,k) = addnoise_spd_de(Y0(:,:,i),noise_size,max_noise);
    end
    for i = (npairs - noutliers+1):npairs
        Ysample(:,:,i,k) = addnoise_spd_de(Y0_outlier(:,:,i),noise_size,max_noise);
    end
end

assert(isspd_mxstack(Ysample) == 1) % 다 spd 인지 확인

Y_raw = Ysample;
