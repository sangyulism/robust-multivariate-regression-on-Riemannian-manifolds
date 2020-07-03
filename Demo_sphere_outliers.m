% DEMO on the unit sphere.
clear;
addpath(genpath(pwd));
rng(222)
disp('Start.');

for iter = 1:30

    mse_p_L2 = 0;
    mse_p_Huber = 0;
    mse_p_L1 = 0;

    mse_v_L2 = 0;
    mse_v_Huber = 0;
    mse_v_L1 = 0;

    seed = randi(1000);

    synth_sphere_data_outliers

    for i = 1:mse_iter
        Y = Y_raw(:,:,i);
        Ybar = karcher_mean_sphere(Y,[],500);   

        [pL2, VL2, EL2, YhatL2, gnormL2] = rmm_sphere_L2(X,Y);
        mse_p_L2 = mse_p_L2 + norm(logmap_sphere(Yp(:,1),pL2))^2;
        VL2_par = paralleltranslateAtoB_sphere(pL2, Yp(:,1),VL2);
        for j = 1:npivots
            mse_v_L2 = mse_v_L2 + norm(V(:,j)-VL2_par(:,j))^2;
        end

        [pHuber, VHuber, EHuber, YhatHuber, gnormHuber] = rmm_sphere_Huber(X,Y);
        mse_p_Huber = mse_p_Huber + norm(logmap_sphere(Yp(:,1),pHuber))^2;
        VHuber_par = paralleltranslateAtoB_sphere(pHuber, Yp(:,1),VHuber);
        for j = 1:npivots
            mse_v_Huber = mse_v_Huber + norm(V(:,j)-VHuber_par(:,j))^2;
        end
        
        [pL1, VL1, EL1, YhatL1, gnormL1] = rmm_sphere_L1(X,Y);
        mse_p_L1 = mse_p_L1 + norm(logmap_sphere(Yp(:,1),pL1))^2;
        VL1_par = paralleltranslateAtoB_sphere(pL1, Yp(:,1),VL1);
        for j = 1:npivots
            mse_v_L1 = mse_v_L1 + norm(V(:,j)-VL1_par(:,j))^2;
        end
    end
    fprintf('%f,%f,%f,', mse_p_L2/mse_iter, mse_p_Huber/mse_iter, mse_p_L1/mse_iter)
    fprintf('%f,%f,%f\n', mse_v_L2/mse_iter, mse_v_Huber/mse_iter, mse_v_L1/mse_iter)
end