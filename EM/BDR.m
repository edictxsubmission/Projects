function [PoE] = BDR(C,dim,Features64,X_BG,X_FG,mu_BG,mu_FG,S_BG,S_FG,pi_BG,pi_FG,mask)

Prior_BG = size(X_BG,1)/(size(X_BG,1) + size(X_FG,1));
Prior_FG = 1 - Prior_BG;

fprintf('\n')
disp('Commence BDR')

M = zeros(size(Features64,1),size(Features64,2));
for y = 1:size(Features64,1)
    for x = 1:size(Features64,2)
        X = Features64{y,x};
        A = 0;
        B = 0;
        for l = 1:C
            A = A + mvnpdf(X(1:dim,:)',mu_BG(l,1:dim),S_BG(1:dim,1:dim,l)) * pi_BG(l,:);
            B = B + mvnpdf(X(1:dim,:)',mu_FG(l,1:dim),S_FG(1:dim,1:dim,l)) * pi_FG(l,:);
        end
        alpha = log(A) + log(Prior_BG);
        beta = log(B) + log(Prior_FG);
        
        if alpha < beta
           M(y,x) = 1;
        end
    end
end

%Compute PoE
E = M-mask;

c_E_0 = 0;
c_E_1 = 0;
for y = 1:size(Features64,1)
    for x = 1:size(Features64,2)
        if E(y,x) < 0
            c_E_1 = c_E_1 + 1;
        elseif E(y,x) > 0
            c_E_0 = c_E_0 + 1;
        end
    end
end

PE_0 = c_E_0/nnz(~mask);
PE_1 = c_E_1/nnz(mask);

PoE = PE_0*Prior_BG + PE_1*Prior_FG; 
disp(['Dimension ', num2str(dim), ' Prob. Error: %', num2str(100*PoE)])

imshow(M,[])
