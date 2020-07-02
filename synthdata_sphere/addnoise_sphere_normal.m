function Anew = addnoise_sphere(A, err_scale, maxerr)
% 길이 dim(Y)(=15) 짜리 unit vector를 받아
    n = size(A,1);
    V = randn(size(A)); %randn(n,m) 은 표준정규분포 난수 n*m 행렬, size(A)는 (15,1) 반환, 즉 여기서 V 는 N(0,1)로 이루어진 15 * 1 벡터.
    V = V-V'*A*A; % T_A(M)으로 V를 projection, 즉 V에서 A와 수직인 성분 분해한 거임.
    V = sphere_randn(n,err_scale,maxerr) * V / norm(V);
    Anew = expmap_sphere(A, V);
end

% f(k) ~ exp(-k^2/2c^2) * sin(k)^(n-2) (0 < k < maxerr < pi) and 0 (o.w) 를 만족하게 x 뽑음
% 이는 f(point)~ exp(-k^2/2c^2) 가 되기 위함임.
% 근데 사실 t분포처럼 해도 될듯.. M 계산 어렵게 할 것 없이.
function x = sphere_randn(n,c,maxerr)
    M = exp(-(n-2)/2) * (n-2)^((n-2)/2) * c^(n-2);
    while true
        k = rand * maxerr;
        if rand < exp(-k^2/(2*c^2)) * sin(k)^(n-2) / M
            x = k;
            break
        end
    end
end

% 밑은 구 버전
% function x = sphere_randn(n,c,maxerr)
%     while true
%         while true
%             k = abs(randn * c);
%             if k < maxerr
%                 break
%             end
%         end
%         if rand < sin(k)^(n-2)
%             x = k;
%             break
%         end
%     end
% end

    
