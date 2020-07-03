% DEMO on SPD manifolds
clear;
addpath(genpath(pwd))
rng(222);
disp('Start.')

iter = 0;

while iter < 10
    var_break = 0;

    mse_p_L2 = 0;
    mse_p_Huber = 0;
    mse_p_L1 = 0;

    mse_v_L2 = 0;
    mse_v_Huber = 0;
    mse_v_L1 = 0;

    seed = randi(1000);

    synth_dti_data_normal

    for i = 1:mse_iter
        Y = Y_raw(:,:,:,i);
        Ybar = karcher_mean_spd(Y,[],500);   

        [pL2, VL2, EL2, YhatL2, gnormL2,Numerical_error] = rmm_spd_L2(X,Y);
        if Numerical_error ==1
            var_break = 1;
            break
        end
        mse_p_L2 = mse_p_L2 + norm_TpM_spd(Yp(:,:,1),logmap_spd(Yp(:,:,1),pL2))^2;
        VL2_par = paralleltranslateAtoB_spd(pL2, Yp(:,:,1),VL2);
        for j = 1:npivots
            mse_v_L2 = mse_v_L2 + norm_TpM_spd(Yp(:,:,1),V(:,:,j)-VL2_par(:,:,j))^2;
        end

        [pHuber, VHuber, EHuber, YhatHuber, gnormHuber,Numerical_error] = rmm_spd_Huber(X,Y);
        if Numerical_error ==1
            var_break = 1;
            break
        end
        mse_p_Huber = mse_p_Huber + norm_TpM_spd(Yp(:,:,1),logmap_spd(Yp(:,:,1),pHuber))^2;
        VHuber_par = paralleltranslateAtoB_spd(pHuber, Yp(:,:,1),VHuber);
        for j = 1:npivots
            mse_v_Huber = mse_v_Huber + norm_TpM_spd(Yp(:,:,1),V(:,:,j)-VHuber_par(:,:,j))^2;
        end
         
        [pL1, VL1, EL1, YhatL1, gnormL1,Numerical_error] = rmm_spd_L1(X,Y);
        if Numerical_error ==1
            var_break = 1;
            break
        end
        mse_p_L1 = mse_p_L1 + norm_TpM_spd(Yp(:,:,1),logmap_spd(Yp(:,:,1),pL1))^2;
        VL1_par = paralleltranslateAtoB_spd(pL1, Yp(:,:,1),VL1);
        for j = 1:npivots
            mse_v_L1 = mse_v_L1 + norm_TpM_spd(Yp(:,:,1),V(:,:,j)-VL1_par(:,:,j))^2;
        end
    end
    if var_break == 1
        continue
    end
    iter = iter + 1;
    fprintf('%f,%f,%f,', mse_p_L2/mse_iter, mse_p_Huber/mse_iter, mse_p_L1/mse_iter)
    fprintf('%f,%f,%f\n', mse_v_L2/mse_iter, mse_v_Huber/mse_iter, mse_v_L1/mse_iter)
end
