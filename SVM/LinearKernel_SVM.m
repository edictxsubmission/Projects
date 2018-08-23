addpath('libsvm-3.22/matlab');
[Label_Test,Label_Train,X_Train,X_Test,Y_Test] = Data();

cost = [2,4,8];

Max_Lagrange = zeros(784,3,10,3);
Min_Lagrange = zeros(784,3,10,3);

margins = zeros(10,20000,3);
accuracy = zeros(3,10);
n_SV = zeros(3,10);
g = zeros(10,10000);
overall_acc = zeros(1,3);

for j = 1:3
    
svmopts = ['-c ',num2str(cost(j)) ' -t 0'];

    for i = 1:10

        disp(['Cost ',num2str(cost(j)) ' - Start Iteration ',num2str(i)])
        tic
        model = svmtrain(Label_Train(:,i), X_Train, svmopts);
        [~,a,~] = svmpredict(Label_Test(:,i),X_Test,model);
        accuracy(j,i) = a(1);
        toc
        %Store total number of support vectors
        n_SV(j,i) = model.totalSV;

        %Find the three largest support vectors of each side
        for ii = 1:3
            [~,Imax] = max(model.sv_coef); 
            Max_Lagrange(:,ii,i,j) = model.SVs(Imax,:)';
            model.sv_coef(Imax) = 0;

            [~,Imin] = min(model.sv_coef); 
            Min_Lagrange(:,ii,i,j) = model.SVs(Imin,:)';
            model.sv_coef(Imin) = 0;
        end

        % Solve for optimal w* (normal vector to hyperplane that minimizes margin)
        w_optimal = model.SVs' * model.sv_coef;

        %Find g where every row has the g(x_test) value of every test sample
        g(i,:) = w_optimal'*X_Test' - model.rho*ones(1,10000); 
        g_train = w_optimal'*X_Train' - model.rho*ones(1,20000);
        margins(i,:,j) = g_train .* transpose(Label_Train(:,i)); 

    end
    
    %Overall Classification
    [~,argmax] = max(g);
    argmax = argmax - 1;
    
    %Overall Classification Error
    err = Y_Test' - argmax;
    overall_acc(j) = nnz(~err)/10000;

end

%Plots###################################################################
for j = 1:3
    figure
    for i = 1:10
        for k = 1:3
            subplot(10,3,3*(i-1) + k)
            maxLang = Max_Lagrange(:,k,i,j);
            maxLang = reshape(maxLang,[28,28]);
            imagesc(maxLang')
        end
    end
end

for j = 1:3
    figure
    for i = 1:10
        for k = 1:3
            subplot(10,3,3*(i-1) + k)
            minLang = Min_Lagrange(:,k,i,j);
            minLang = reshape(minLang,[28,28]);
            imagesc(minLang')
        end
    end
end


%CDF plots 

for j = 1:3
    figure 
    for i = 1:10
        subplot(4,3,i)
        cdfplot(margins(i,:,j)')
        title(['Class ', num2str(i-1)])
    end
end

