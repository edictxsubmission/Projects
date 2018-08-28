
# coding: utf-8

# In[1]:


# !pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl 
# !pip3 install torchvision
# !pip install tqdm


# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
import torch.nn.init as init
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from random import shuffle
import matplotlib.pyplot as plt
import math


# In[2]:


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
valid_size=0.9
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform);
num_train = len(trainset);
indices = list(range(num_train));
split = int(np.floor(valid_size * num_train));
np.random.shuffle(indices);
train_idx, valid_idx = indices[:split], indices[split:];
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,sampler=train_sampler, num_workers=1);
validloader = torch.utils.data.DataLoader(trainset, batch_size=50,sampler=valid_sampler, num_workers=1);

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform);
testloader = torch.utils.data.DataLoader(testset, batch_size=50,shuffle=True, num_workers=1);

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck');


# In[3]:


class Net(nn.Module):

    def __init__(self,kernel,channels,dim):
        super(Net, self).__init__()
        f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13 = kernel;
        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13 = channels;
        
        self.conv1 = nn.Conv2d(3, c1, f1,bias =True,padding = 1);
        init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2.0));
        init.constant(self.conv1.bias, 0.1);
        self.batch_conv1 = nn.BatchNorm2d(c1);
        #dim = dim - f1 +1;
        
        self.conv2 = nn.Conv2d(c1, c2, f2,bias =True,padding = 1);
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0));
        init.constant(self.conv2.bias, 0.1);
        self.batch_conv2 = nn.BatchNorm2d(c2);
        #dim = math.floor((dim - f2)/2) +1;
        
        self.maxpool1 = nn.MaxPool2d((2,2),stride=2)
        dim = int(((dim)/2.0) )
        
        self.conv3 = nn.Conv2d(c2, c3, f3,bias =True,padding = 1);
        init.xavier_uniform(self.conv3.weight, gain=np.sqrt(2.0));
        init.constant(self.conv3.bias, 0.1);
        self.batch_conv3 = nn.BatchNorm2d(c3);
        #dim = dim - f3 +1;
        
        self.conv4 = nn.Conv2d(c3, c4, f4,bias =True,padding = 1);
        init.xavier_uniform(self.conv4.weight, gain=np.sqrt(2.0));
        init.constant(self.conv4.bias, 0.1);
        self.batch_conv4 = nn.BatchNorm2d(c4);
        #dim = int(math.floor((dim - f4)/2) +1);
        
        self.maxpool2 = nn.MaxPool2d((2,2),stride=2)
        dim = int(((dim)/2.0))
        
        self.conv5 = nn.Conv2d(c4, c5, f5,bias =True,padding = 1);
        init.xavier_uniform(self.conv5.weight, gain=np.sqrt(2.0));
        init.constant(self.conv5.bias, 0.1);
        self.batch_conv5 = nn.BatchNorm2d(c5);
        #dim = int(dim - f5 +1);
        
        self.conv6 = nn.Conv2d(c5, c6, f6,bias =True,padding = 1);
        init.xavier_uniform(self.conv6.weight, gain=np.sqrt(2.0));
        init.constant(self.conv6.bias, 0.1);
        self.batch_conv6 = nn.BatchNorm2d(c6);
        #dim = int(dim - f5 +1);
        
        self.conv7 = nn.Conv2d(c6, c7, f7,bias =True,padding = 1);
        init.xavier_uniform(self.conv7.weight, gain=np.sqrt(2.0));
        init.constant(self.conv7.bias, 0.1);
        self.batch_conv7 = nn.BatchNorm2d(c7);
        #dim = int(dim - f5 +1);
        
        self.maxpool3 = nn.MaxPool2d((2,2),stride=2)
        dim = int(((dim)/2.0))
        
        self.conv8 = nn.Conv2d(c7, c8, f8,bias =True,padding = 1);
        init.xavier_uniform(self.conv8.weight, gain=np.sqrt(2.0));
        init.constant(self.conv8.bias, 0.1);
        self.batch_conv8 = nn.BatchNorm2d(c8);
        #dim = int(dim - f5 +1);
        
        self.conv9 = nn.Conv2d(c8, c9, f9,bias =True,padding = 1);
        init.xavier_uniform(self.conv9.weight, gain=np.sqrt(2.0));
        init.constant(self.conv9.bias, 0.1);
        self.batch_conv9 = nn.BatchNorm2d(c9);
        #dim = int(dim - f5 +1);
        
        self.conv10 = nn.Conv2d(c9, c10, f10,bias =True,padding = 1);
        init.xavier_uniform(self.conv10.weight, gain=np.sqrt(2.0));
        init.constant(self.conv10.bias, 0.1);
        self.batch_conv10 = nn.BatchNorm2d(c10);
        #dim = int(dim - f5 +1);
               
        self.maxpool4 = nn.MaxPool2d((2,2),stride=2)
        dim = int(((dim)/2.0))
        
        self.conv11 = nn.Conv2d(c10, c11, f11,bias =True,padding = 1);
        init.xavier_uniform(self.conv11.weight, gain=np.sqrt(2.0));
        init.constant(self.conv11.bias, 0.1);
        self.batch_conv11 = nn.BatchNorm2d(c11);
        #dim = int(dim - f5 +1);
        
        self.conv12 = nn.Conv2d(c11, c12, f12,bias =True,padding = 1);
        init.xavier_uniform(self.conv12.weight, gain=np.sqrt(2.0));
        init.constant(self.conv12.bias, 0.1);
        self.batch_conv12 = nn.BatchNorm2d(c12);
        #dim = int(dim - f5 +1);
        
        self.conv13 = nn.Conv2d(c12, c13, f13,bias =True,padding = 1);
        init.xavier_uniform(self.conv13.weight, gain=np.sqrt(2.0));
        init.constant(self.conv13.bias, 0.1);
        self.batch_conv13 = nn.BatchNorm2d(c13);
        
        # an affine operation: y = Wx + b
        
        self.fc1 = nn.Linear(c13 * dim * dim, 800)
        self.fc2 = nn.Linear(800, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.batch_conv1(self.conv1(x)))
        x = F.relu(self.batch_conv2(self.conv2(x)))
        x = self.maxpool1(x)
        
        x = F.relu(self.batch_conv3(self.conv3(x)))
        x = F.relu(self.batch_conv4(self.conv4(x)))
        x = self.maxpool2(x)
        
        x = F.relu(self.batch_conv5(self.conv5(x)))
        x = F.relu(self.batch_conv6(self.conv6(x)))
        x = F.relu(self.batch_conv7(self.conv7(x)))
        x = self.maxpool3(x)
        
        x = F.relu(self.batch_conv8(self.conv8(x)))
        x = F.relu(self.batch_conv9(self.conv9(x)))
        x = F.relu(self.batch_conv10(self.conv10(x))) 
        x = self.maxpool4(x)
        
        x = F.relu(self.batch_conv11(self.conv11(x)))
        x = F.relu(self.batch_conv12(self.conv12(x)))
        x = F.relu(self.batch_conv13(self.conv13(x))) 
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[4]:


dim=int(32);
filters = (3,3,3,3,3,3,3,3,3,3,3,3,3);
channels = (64,64,128,128,256,256,256,512,512,512,512,512,512);
model = Net(filters,channels,dim);

model.cuda();
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# In[5]:


train_entropy = list();
valid_entropy = list();
test_entropy = list();
train_accuracy =list();
test_accuracy = list();
valid_accuracy = list();
for epoch in trange(20):
    model.train();
    error_train =0;
    total_train=0;
    correct_train=0;
    for i, (images, labels) in enumerate(trainloader):
        images = Variable(images).cuda()
        label = Variable(labels).cuda()        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, label)
        error_train+=loss.data[0];
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted.cpu() == labels).sum()
        total_train += labels.size(0)
        loss.backward()
        optimizer.step()
        del images
        del labels
    train_entropy.append(error_train/float(i));
    train_accuracy.append((correct_train/float(total_train))*100);
    model.eval();
    error_valid=0;
    total_valid=0;
    correct_valid=0;
    for i, (images, labels) in enumerate(validloader):
        images = Variable(images).cuda()
        label = Variable(labels).cuda()        
        # Forward + Backward + Optimize
        outputs = model(images)
        loss_valid = criterion(outputs, label)
        error_valid+=loss_valid.data[0];
        _, predicted_valid = torch.max(outputs.data, 1)
        correct_valid += (predicted_valid.cpu() == labels).sum()
        total_valid += labels.size(0);
        del images
        del labels
    valid_entropy.append(error_valid/float(i));
    valid_accuracy.append((correct_valid/float(total_valid))*100);
    
    error_test=0;
    total_test=0;
    correct_test=0;
    for i, (images, labels) in enumerate(testloader):
        images = Variable(images).cuda()
        label = Variable(labels).cuda()        
        # Forward + Backward + Optimize
        outputs = model(images)
        loss_test = criterion(outputs, label)
        error_test+=loss_test.data[0];
        _, predicted_test = torch.max(outputs.data, 1)
        correct_test += (predicted_test.cpu() == labels).sum()
        total_test += labels.size(0);
        del images
        del labels
    test_entropy.append(error_test/float(i));
    test_accuracy.append((correct_test/float(total_test))*100);


# In[6]:


model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in testloader:
    images = Variable(images).cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))


# In[7]:


model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in trainloader:
    images = Variable(images).cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Train Accuracy of the model on the 50000 train images: %d %%' % (100 * correct / total))


# In[12]:


plt.plot(train_entropy,label="Training Entropy");
plt.plot(valid_entropy,label="Validation Entropy");
plt.plot(test_entropy,label="Test Entropy");
plt.legend()
plt.ylabel('Entropy')
plt.xlabel('Epochs')
plt.show()


# In[14]:


plt.plot(train_accuracy,label="Training Accuracy");
plt.plot(valid_accuracy,label="Validation Accuracy");
plt.plot(test_accuracy,label="Test Accuracy")
plt.legend();
plt.ylabel('Accuracy (%)')
plt.xlabel('Epochs')
plt.show()


# In[11]:


#train_accuracy

