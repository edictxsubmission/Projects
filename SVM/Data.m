function [Label_Test,Label_Train,X_Train,X_Test,Y_Test] = Data()

load('Data.mat')
X_Train = training_images(1:20000,:);
X_Test = test_images;
Y_Train = training_labels(1:20000,:);
Y_Test = test_labels;

%1a) Produce 10 Training and Testing Labels Vectors 
Label_Train = -1*ones(20000,10);
Label_Test = -1*ones(10000,10);

for i = 1:20000
   Label_Train(i,Y_Train(i)+1) = 1;
end

for i = 1:10000
   Label_Test(i,Y_Test(i)+1) = 1;
end

