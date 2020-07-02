% DEMO on the unit sphere.
clear;
addpath(genpath(pwd));
rng(222)
disp('Start.');

for iter = 1:2

    mse_p_ls = 0;
    mse_p_hb = 0;
    mse_p_l1 = 0;

    mse_v_ls = 0;
    mse_v_hb = 0;
    mse_v_l1 = 0;

    seed = randi(1000);

    synth_sphere_data_de

    for i = 1:mse_iter
        Y = Y_raw(:,:,i);
        Ybar = karcher_mean_sphere(Y,[],500);   

        [pls, Vls, Els, Yhatls, gnormls,gradp_list] = mglm_sphere(X,Y);
        mse_p_ls = mse_p_ls + norm(logmap_sphere(Yp(:,1),pls))^2;
        Vls_par = paralleltranslateAtoB_sphere(pls, Yp(:,1),Vls);
        for j = 1:npivots
            mse_v_ls = mse_v_ls + norm(V(:,j)-Vls_par(:,j))^2;
        end

        [phb, Vhb, Ehb, Yhathb, gnormhb] = mglm_sphere_huber(X,Y);
        mse_p_hb = mse_p_hb + norm(logmap_sphere(Yp(:,1),phb))^2;
        Vhb_par = paralleltranslateAtoB_sphere(phb, Yp(:,1),Vhb);
        for j = 1:npivots
            mse_v_hb = mse_v_hb + norm(V(:,j)-Vhb_par(:,j))^2;
        end
        
        [pl1, Vl1, El1, Yhatl1, gnorml1] = mglm_sphere_l1(X,Y);
        mse_p_l1 = mse_p_l1 + norm(logmap_sphere(Yp(:,1),pl1))^2;
        Vl1_par = paralleltranslateAtoB_sphere(pl1, Yp(:,1),Vl1);
        for j = 1:npivots
            mse_v_l1 = mse_v_l1 + norm(V(:,j)-Vl1_par(:,j))^2;
        end
    end
    fprintf('%f,%f,%f,', mse_p_ls/mse_iter, mse_p_hb/mse_iter, mse_p_l1/mse_iter)
    fprintf('%f,%f,%f\n', mse_v_ls/mse_iter, mse_v_hb/mse_iter, mse_v_l1/mse_iter)
end