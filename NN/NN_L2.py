#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:27:13 2018

@author: eliuc
"""
import numpy as np
from readData import read_idx
import matplotlib.pyplot as plt

#Process Data Input
Testing_Images = read_idx('t10k-images.idx3-ubyte')
Testing_Labels = read_idx('t10k-labels.idx1-ubyte')
Training_Images = read_idx('train-images.idx3-ubyte')
Training_Labels = read_idx('train-labels.idx1-ubyte')

#train_n = 3000
#test_n = 1000
train_n = Training_Images.shape[0]
test_n = Testing_Images.shape[0]

X_Images = np.zeros((28*28,train_n))
Y_Labels = np.reshape(Training_Labels[0:train_n],(1,train_n))
for i in range(0,train_n):
    X_Images[:,i] = np.squeeze(np.reshape(Training_Images[i,:,:],(28*28,1)))
X_Images = (X_Images/127.5) - np.ones((28*28,1))

#Testing Data
X_Testing = np.zeros((28*28,test_n))
Y_Testing = np.reshape(Testing_Labels[0:test_n],(1,test_n))
for i in range(0,test_n):
    X_Testing[:,i] = np.squeeze(np.reshape(Testing_Images[i,:,:],(28*28,1)))
X_Testing = (X_Testing/127.5) - np.ones((28*28,1))

#Set dimensional parameters
dx, m = X_Images.shape 
dx, mtest = X_Testing.shape 
mval = int(.1*m)
mtrain = int(.9*m)

#Set Scalar Parameters
H1, H2, C, ITER, Trigger = 64, 64, 10,  0, 0
Li_Train, Li_Test, Li_Validation = 0,0,0
eta = .6
mbatch = 128


#Separate Validation from Training 
X_Training = X_Images[:,0:mtrain]
Y_Training = Y_Labels[:,0:mtrain]

X_Validation = X_Images[:,mtrain:mtrain+mval]
Y_Validation = Y_Labels[:,mtrain:mtrain+mval]

Y_Training_1Hot = np.zeros((C,mtrain))
Y_Testing_1Hot = np.zeros((C,mtest))
Y_Validation_1Hot = np.zeros((C,mval))

Eye = np.identity(10)

for i in range(0,mtrain):
    Y_Training_1Hot[:,i] = np.squeeze(Eye[:,int(Y_Training[:,i])]) 

for i in range(0,mval):
    Y_Validation_1Hot[:,i] = np.squeeze(Eye[:,int(Y_Validation[:,i])]) 
    
for i in range(0,int(mtest)):
    Y_Testing_1Hot[:,i] = np.squeeze(Eye[:,int(Y_Testing[:,i])]) 

#Initialize Lists 
Loss_Train = []
Loss_Test = []
Loss_Validation = []
Acc_Train = []
Acc_Test = []
Acc_Validation = []

#Weight Initialization 
W1 = np.random.rand(H1,dx)*.01
W2 = np.random.rand(H2,H1)*.01
W3 = np.random.rand(C,H2)*.01
b1 = np.ones((H1,1))
b2 = np.ones((H2,1))
b3 = np.ones((C,1))

#Functions 

def sigmoid(z):
    return 1/(1+np.exp(-z))

def dsigmoid(zz):
    return sigmoid(zz)*(1-sigmoid(zz))

def P(zzz):
    #Input EE is a nx1 vector
    EE = np.exp(zzz)
    S = np.sum(EE,axis=0,dtype=np.float)
    F = EE / S
    return F

#Iteration Begins
cyc = int(mtrain)/128

while Trigger == 0:
#while ITER < 2000:
    for ii in range(0,cyc):
        X_Training_mb = X_Training[:,128*ii:128*(ii+1)]
        
        #Traning A1 and A2
        Z1_Training_mb = np.dot(W1,X_Training_mb)+b1
        A1_Training_mb = sigmoid(Z1_Training_mb)
        Z2_Training_mb = np.dot(W2,A1_Training_mb)+b2
        A2_Training_mb = sigmoid(Z2_Training_mb)    
        Z3_Training_mb = np.dot(W3,A2_Training_mb)+b3
        A3_Training_mb = P(Z3_Training_mb)

        #Update Gradient Descent Weights
        dZ3 = A3_Training_mb - Y_Training_1Hot[:,128*ii:128*(ii+1)]
        dW3 = (1/float(mbatch))*np.dot(dZ3,A2_Training_mb.T)
        
        dZ2 = np.dot(W3.T,dZ3) * dsigmoid(Z2_Training_mb)
        dW2 = (1/float(mbatch))*np.dot(dZ2,A1_Training_mb.T)
        
        dZ1 = np.dot(W2.T,dZ2) * dsigmoid(Z1_Training_mb)
        dW1 = (1/float(mbatch))*np.dot(dZ1,X_Training_mb.T)
        
        db1 = (1/float(mbatch))*np.sum(dZ1,axis=1,keepdims=1)
        db2 = (1/float(mbatch))*np.sum(dZ2,axis=1,keepdims=1)
        db3 = (1/float(mbatch))*np.sum(dZ3,axis=1,keepdims=1)

        #Gradient Descent Paramater Update
        
        W1 = W1 - eta*dW1
        W2 = W2 - eta*dW2
        W3 = W3 - eta*dW3
        b1 = b1 - eta*db1
        b2 = b2 - eta*db2
        b3 = b3 - eta*db3
        
    #Traning A1 and A2 and A3
    Z1_Training = np.dot(W1,X_Training)+b1
    A1_Training = sigmoid(Z1_Training)
    Z2_Training = np.dot(W2,A1_Training)+b2
    A2_Training = sigmoid(Z2_Training)
    Z3_Training = np.dot(W3,A2_Training)+b3
    A3_Training = P(Z3_Training)
    
    #Testing A1 and A2 and A3
    Z1_Testing = np.dot(W1,X_Testing)+b1
    A1_Testing = sigmoid(Z1_Testing)
    Z2_Testing = np.dot(W2,A1_Testing)+b2
    A2_Testing = sigmoid(Z2_Testing)
    Z3_Testing = np.dot(W3,A2_Testing)+b3
    A3_Testing = P(Z3_Testing)
    
    #Validation A1 and A2 and A3
    Z1_Validation = np.dot(W1,X_Validation)+b1
    A1_Validation = sigmoid(Z1_Validation)
    Z2_Validation = np.dot(W2,A1_Validation)+b2
    A2_Validation = sigmoid(Z2_Validation)
    Z3_Validation = np.dot(W3,A2_Validation)+b3
    A3_Validation = P(Z3_Validation)
    
    #Softmax Loss for Training, Testing, and Validation Sets

    Li_Train = (1/float(mtrain))*sum(sum(np.log(A3_Training)*Y_Training_1Hot))
    Loss_Train.append(-1*Li_Train) 
    
    Li_Test = (1/float(mtest))*sum(sum(np.log(A3_Testing)*Y_Testing_1Hot)) 
    Loss_Test.append(-1*Li_Test) 
    
    Li_Validation = (1/float(mval))*sum(sum(np.log(A3_Validation)*Y_Validation_1Hot)) 
    Loss_Validation.append(-1*Li_Validation)

    #Find Accuracy for Training, Testing, and Validation Sets
    Y_NN_Training = np.argmax(A3_Training, axis=0)
    E_Training = Y_NN_Training - Y_Training 
    accuracyTraining = 1 - (np.count_nonzero(E_Training)/float(E_Training.shape[1]))
    Acc_Train.append(accuracyTraining)
    
    Y_NN_Testing = np.argmax(A3_Testing, axis=0)
    E_Testing = Y_NN_Testing - Y_Testing
    accuracyTesting = 1 - (np.count_nonzero(E_Testing)/float(E_Testing.shape[1]))
    Acc_Test.append(accuracyTesting)
    
    Y_NN_Validation = np.argmax(A3_Validation, axis=0)
    E_Validation = Y_NN_Validation - Y_Validation
    accuracyValidation = 1 - (np.count_nonzero(E_Validation)/float(E_Validation.shape[1]))
    Acc_Validation.append(accuracyValidation)
    

    #Dermine Threshold g to assure early stopping starts after the 4th iteration
    g = len(Loss_Validation)

    #Exit Condition
    if g >= 30:   
       if Loss_Validation[ITER] > Loss_Validation[ITER - 1] and Loss_Validation[ITER - 1] > Loss_Validation[ITER - 2] and Loss_Validation[ITER - 2] > Loss_Validation[ITER - 3]:
           Trigger = 1

    print "Validati Acc:", Acc_Validation[ITER], " Validati Loss: ", Loss_Validation[ITER]
    print "Training Acc:", Acc_Train[ITER], " Training Loss: ", Loss_Train[ITER]
    print "Testing  Acc:", Acc_Test[ITER], " Testing  Loss: ", Loss_Test[ITER], "\n"

    ITER = ITER + 1


TestLoss = np.asarray(Loss_Test)
TrainLoss = np.asarray(Loss_Train)
ValidationLoss = np.asarray(Loss_Validation)

AccuracyTrain = np.asarray(Acc_Train)
AccuracyTest = np.asarray(Acc_Test)
AccuracyValidation = np.asarray(Acc_Validation)

x = np.arange(ITER)

#Plot Accuracy
plt.plot(x, AccuracyTrain, label='Training')
plt.plot(x, AccuracyTest, label='Testing')
plt.plot(x, AccuracyValidation, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title("Accuracy")
plt.legend()
#plt.savefig('3Layer_Accuracy', bbox_inches='tight')
plt.show()
#plt.close
#Plot Loss

plt.plot(x, TrainLoss, label='Training')
plt.plot(x, TestLoss, label='Testing')
plt.plot(x, ValidationLoss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss")
plt.legend()
#plt.savefig('3Layer_Loss', bbox_inches='tight')
plt.show()