function [mu_out, S_out, pi_out] = EM(X,C)

% Input: Dataset with each sample being in a different row 
% X(i,:), as well as parameters of each Gaussian component mu(i,:) (mean), 
% covariance S(:,:,i), and weight pi(i,:).

% Output: Updated parameters of each Gaussian component mu(i,:) (mean), 
% covariance S(:,:,i), and weight pi(i,:).

D = size(X,2); % Dataset Dimension  
N = size(X,1); % Dataset Samples

%Randomized Initialization Parameters
pi = rand(C,1);
mu = 10*rand(C,D) - 5;
S = zeros(D,D,C);

for i = 1:C
    S(:,:,i) = diag(5*(rand(1,D))+7.5);
end

%Find Initial Log-Likelihood
LogL = 0;
for i=1:N
    A=0;
    for j=1:C
        A = A + mvnpdf(X(i,:),mu(j,:),S(:,:,j))*pi(j,:);
    end
    LogL = LogL + log(A);
end

Likelihood_D(:,1) = LogL;
iter = 1;
LogL = inf;
prev_LogL = 0;

H = zeros(N,C);       
           
fprintf('\n')
disp('Commence EM Process')

%EM 
while abs(LogL - prev_LogL)/prev_LogL > 0.001  || iter <= 4 
    
    iter = iter + 1;
    prev_LogL = LogL;
    
    %Expectation Step
    for a = 1:N
        den1 = 0;
        for aa = 1:C
            den1 = den1 + mvnpdf(X(a,:),mu(aa,:),S(:,:,aa))*pi(aa);
        end
        for b = 1:C
            num1 = mvnpdf(X(a,:),mu(b,:),S(:,:,b))*pi(b);
            if isnan(num1/den1)
                H(a,b) = 1;
            else
                H(a,b) = num1/den1;
            end
        end
    end
    
    %Maximization Step
    for c = 1:C
        v = ((X' - mu(c,:)'*ones(1,N)).^2)*H(:,c) / sum(H(:,c));
        t = isnan(v);
        v(t) = 0.00001;
        z = find(~v,1);
        if ~isempty(z)
            disp('Underflow Correction')
            v(v < 0.00001) = 0.00001;
        end
        v(v < 0.00001) = 0.00001;
        S(:,:,c) = diag(v');
        mu(c,:) = (X'*H(:,c)) / sum(H(:,c));
        pi(c) = (1/N)*sum(H(:,c));
    end

    if iter == 75
        disp('No Convergence')
        break
    end

    %Compute Log-Likelihood
    
    LogL=0;
    for i=1:N
        A=0;
        for j=1:C       
            A = A + mvnpdf(X(i,:),mu(j,:),S(:,:,j))*pi(j,:);     
        end
        LogL = LogL + log(A);
    end
    
    Likelihood_D(:,iter) = LogL;
    disp(['Iteration ',num2str(iter)])
    disp(['Log-Likelihood: ',num2str(LogL)])
    disp(['Log-Likelihood Change: %',num2str(100*abs(LogL - prev_LogL)/prev_LogL)])
    
end

mu_out = mu;
S_out = S;
pi_out = pi;
L = Likelihood_D;