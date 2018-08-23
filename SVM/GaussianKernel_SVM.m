addpath('libsvm-3.22/matlab');
[Label_Test,Label_Train,X_Train,X_Test,Y_Test] = Data();

Max_Lagrange = zeros(784,3,10);
Min_Lagrange = zeros(784,3,10);

margins = zeros(10,20000);

gamma = 0.0625;
cost = 2;

accuracy = zeros(1,10);
svmopts = ['-c ', num2str(cost) ' -t 2 -g ', num2str(gamma)];
n_SV = zeros(1,10);
g1 = zeros(10,10000);

for i = 1:10
    
    disp(['Start Class ',num2str(i)])
    tic
    model = svmtrain(Label_Train(:,i), X_Train, svmopts);
    [~,a,~] = svmpredict(Label_Test(:,i),X_Test,model);
    accuracy(i) = a(1);
    toc
    %Store total number of support vectors
    n_SV(i) = model.totalSV;
    
    %Find the three largest support vectors of each side
    for ii = 1:3
        [~,Imax] = max(model.sv_coef); 
        Max_Lagrange(:,ii,i) = model.SVs(Imax,:)';
        model.sv_coef(Imax) = 0;
        
        [~,Imin] = min(model.sv_coef); 
        Min_Lagrange(:,ii,i) = model.SVs(Imin,:)';
        model.sv_coef(Imin) = 0;
    end
    
    a=0;b=0;c=0;d=0;
    %Construct kernelized summation
    wK_test = zeros(1,10000);
    wK_train = zeros(1,20000);
    
    disp(['Commence Kernelization of Class ',num2str(i)])
    
    for j = 1:model.totalSV
        expon_test = exp(-gamma*sum((model.SVs(j,:)'*ones(1,10000) - X_Test').^2));
        expon_train = exp(-gamma*sum((model.SVs(j,:)'*ones(1,20000) - X_Train').^2));

        wKi_test = model.sv_coef(j)*expon_test;
        wKi_train = model.sv_coef(j)*expon_train;

        wK_test = wK_test + wKi_test;
        wK_train = wK_train + wKi_train;
        if (j/model.totalSV) > .25 && (j/model.totalSV) < .27 && a == 0
            a = 1;
            disp('25% Complete')
        end
        if (j/model.totalSV) > .50 && (j/model.totalSV) < .52 && b == 0
            b = 1;
            disp('50% Complete')
        end
        if (j/model.totalSV) > .75 && (j/model.totalSV) < .77 && c == 0
            c = 1;
            disp('75% Complete')
        end
        if (j/model.totalSV) > .95 && (j/model.totalSV) < .97 && d == 0
            d = 1;
            disp('95% Complete')
        end      
    end
    
    %Find g where every row has the g(x_test) value of every test sample
    g1(i,:) = wK_test - model.rho*ones(1,10000); 
    g1_train = wK_train - model.rho*ones(1,20000);
    margins(i,:) = g1_train .* transpose(Label_Train(:,i)); 
    
end

%Overall Classification
[~,argmax1] = max(g1);
argmax1 = argmax1 - 1;

%Overall Classification Error
err1 = Y_Test' - argmax1;
overall_acc1 = nnz(~err1)/10000;

%Plots###################################################################

figure
for i = 1:10
    subplot(10,3,3*(i-1) + 1)
    maxLang = Max_Lagrange(:,1,i);
    maxLang = reshape(maxLang,[28,28]);
    imagesc(maxLang')
    
    subplot(10,3,3*(i-1) + 2)
    maxLang = Max_Lagrange(:,2,i);
    maxLang = reshape(maxLang,[28,28]);
    imagesc(maxLang')
    
    subplot(10,3,3*(i-1) + 3)
    maxLang = Max_Lagrange(:,3,i);
    maxLang = reshape(maxLang,[28,28]);
    imagesc(maxLang')
end
%title('Max Lagranges Support Vectors | C = 2')

figure
for i = 1:10
    subplot(10,3,3*(i-1) + 1)
    minLang = Min_Lagrange(:,1,i);
    minLang = reshape(minLang,[28,28]);
    imagesc(minLang')
    
    subplot(10,3,3*(i-1) + 2)
    minLang = Min_Lagrange(:,2,i);
    minLang = reshape(minLang,[28,28]);
    imagesc(minLang')
    
    subplot(10,3,3*(i-1) + 3)
    minLang = Min_Lagrange(:,3,i);
    minLang = reshape(minLang,[28,28]);
    imagesc(minLang')
end
%title('Min Lagranges Support Vectors | C = 2')

%CDF plots 

figure 
for i = 1:10
    subplot(4,3,i)
    cdfplot(margins(i,:)')
    title(['Class ', num2str(i-1)])
end
