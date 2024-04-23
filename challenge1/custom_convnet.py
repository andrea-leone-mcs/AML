# custom convnet 

import torch 

class CustomModel(torch.nn.Module):

    def __init__(self):
        super(CustomModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.activation1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.activation2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = torch.nn.Conv2d(64, 128, 3)
        self.activation3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(128, 128, 3)
        self.activation4 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(2, 2)

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(128, 6)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(6, 1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.conv2(x)  
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = self.conv4(x)
        x = self.activation4(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

tinymodel = CustomModel()
