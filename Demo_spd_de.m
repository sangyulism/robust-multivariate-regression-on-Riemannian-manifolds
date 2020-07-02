% DEMO on SPD manifolds
clear;
addpath(genpath(pwd))
rng(222);
disp('Start.')

iter = 0;

while iter < 10
    var_break = 0;

    mse_p_ls = 0;
    mse_p_hb = 0;
    mse_p_l1 = 0;

    mse_v_ls = 0;
    mse_v_hb = 0;
    mse_v_l1 = 0;

    seed = randi(1000);

    synth_dti_data_de

    for i = 1:mse_iter
        Y = Y_raw(:,:,:,i);
        Ybar = karcher_mean_spd(Y,[],500);   

        [pls, Vls, Els, Yhatls, gnormls,Numerical_error] = mglm_spd(X,Y);
        if Numerical_error ==1
            var_break = 1;
            break
        end
        mse_p_ls = mse_p_ls + norm_TpM_spd(Yp(:,:,1),logmap_spd(Yp(:,:,1),pls))^2;
        Vls_par = paralleltranslateAtoB_spd(pls, Yp(:,:,1),Vls);
        for j = 1:npivots
            mse_v_ls = mse_v_ls + norm_TpM_spd(Yp(:,:,1),V(:,:,j)-Vls_par(:,:,j))^2;
        end

        [phb, Vhb, Ehb, Yhathb, gnormhb,Numerical_error] = mglm_spd_huber(X,Y);
        if Numerical_error ==1
            var_break = 1;
            break
        end
        mse_p_hb = mse_p_hb + norm_TpM_spd(Yp(:,:,1),logmap_spd(Yp(:,:,1),phb))^2;
        Vhb_par = paralleltranslateAtoB_spd(phb, Yp(:,:,1),Vhb);
        for j = 1:npivots
            mse_v_hb = mse_v_hb + norm_TpM_spd(Yp(:,:,1),V(:,:,j)-Vhb_par(:,:,j))^2;
        end
         
        [pl1, Vl1, El1, Yhatl1, gnorml1,Numerical_error] = mglm_spd_l1(X,Y);
        if Numerical_error ==1
            var_break = 1;
            break
        end
        mse_p_l1 = mse_p_l1 + norm_TpM_spd(Yp(:,:,1),logmap_spd(Yp(:,:,1),pl1))^2;
        Vl1_par = paralleltranslateAtoB_spd(pl1, Yp(:,:,1),Vl1);
        for j = 1:npivots
            mse_v_l1 = mse_v_l1 + norm_TpM_spd(Yp(:,:,1),V(:,:,j)-Vl1_par(:,:,j))^2;
        end
    end
    if var_break == 1
        continue
    end
    iter = iter + 1;
    fprintf('%f,%f,%f,', mse_p_ls/mse_iter, mse_p_hb/mse_iter, mse_p_l1/mse_iter)
    fprintf('%f,%f,%f\n', mse_v_ls/mse_iter, mse_v_hb/mse_iter, mse_v_l1/mse_iter)
end
