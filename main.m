%Get Data 
[X_BG,X_FG,mask,Features64] = Data();

C = 8;
dimension = [1,2,4,8,16,32,40,48,56,64]; % Dimensions 

PoE = zeros(5,10,5);
for r = 1:5
    [mu_BG, S_BG, pi_BG] = EM(X_BG,C);
    for s = 1:5
        [mu_FG, S_FG, pi_FG] = EM(X_FG,C);
        for p = 1:size(dimension,2)
            PoE(s,p,r) = BDR(C,dimension(p),Features64,X_BG,X_FG,mu_BG,mu_FG,S_BG,S_FG,pi_BG,pi_FG,mask);
        end
    end
end
     
for j = 1:5
    figure
    plot(dimension,PoE(1,1:10,j),'r-o')
    hold on
    plot(dimension,PoE(2,1:10,j),'g-o')
    hold on
    plot(dimension,PoE(3,1:10,j),'b-o')
    hold on
    plot(dimension,PoE(4,1:10,j),'y-o')
    hold on
    plot(dimension,PoE(5,1:10,j),'m-o')
    hold off
    title(['Probability of Error vs. Dimension for Randomized BG Init. #',num2str(j)])
    xlabel('Dimension')
    ylabel('PoE')
    legend('FG Init. #1','FG Init. #2','FG Init. #3','FG Init. #4','FG Init. #5')
end


