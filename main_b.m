%Get Data 
[X_BG,X_FG,mask,Features64] = Data();

C = [1,2,4,16,32];
dimension = [1,2,4,8,16,32,40,48,56,64]; % Dimensions 

PoE = zeros(5,10);

for q = 1:size(C,2)
    [mu_BG, S_BG, pi_BG] = EM(X_BG,C(q));
    [mu_FG, S_FG, pi_FG] = EM(X_FG,C(q));
    for p = 1:size(dimension,2)
        PoE(q,p) = BDR(C(q),dimension(p),Features64,X_BG,X_FG,mu_BG,mu_FG,S_BG,S_FG,pi_BG,pi_FG,mask);
    end
end

figure
plot(dimension,PoE(1,:),'r-o')
hold on
plot(dimension,PoE(2,:),'g-o')
hold on
plot(dimension,PoE(3,:),'b-o')
hold on
plot(dimension,PoE(4,:),'y-o')
hold on
plot(dimension,PoE(5,:),'m-o')
hold off
title('Probability of Error vs. Dimension for Varying Gaussian Mixture Components')
xlabel('Dimension')
ylabel('PoE')
legend('C = 1','C = 2','C = 4','C = 16','C = 32')
