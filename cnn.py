import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# using GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# hyperparameters
epochs = 1
classes = 10
batch_size = 256
learning_rate = 0.001

# dataset
train_dataset = torchvision.datasets.MNIST(root='\\data\\',
                                           train=True,
                                           transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))

test_dataset = torchvision.datasets.MNIST(root='\\data\\',
                                           train=False,
                                           transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))

# load data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

# todo 将卷积添加到最后一层，全局池化
# todo 换一个cnn结构

# convolutional neural network
class CNN(nn.Module):
    def __init__(self, classes=10):
        super(CNN, self).__init__()
        # 28 14 ->  32  16
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 14 7  -> 16   8
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 8 4
        self.layer3 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 4 2
        self.layer4 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.fc1 = nn.Linear(2*2*512,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,classes)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

model = CNN(classes).to(device)


# loss and optimizer
criter = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# train
total_step = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criter(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}'.format(epoch+1,epochs,i+1,total_step,loss.item()))

# test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predict = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

print(model)
# save model
torch.save(model.state_dict(),'cnn_test_model.pth')



