import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch import nn
from B2 import extract as get_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_class = 5
epoch = 100

learning_rate = 0.001

x_train, y_train = get_data.get_train_data()
x_train = x_train.reshape(10000, 3, 64, 64)
#print(np.shape(x_train))
train = torch.utils.data.TensorDataset(x_train, y_train)
train_data = DataLoader(train, batch_size=128, shuffle=True)

x_test, y_test = get_data.get_test_data()
x_test = x_test.reshape(2500, 3, 64, 64)
test_set = torch.utils.data.TensorDataset(x_test, y_test)



class Conv_net(nn.Module):
    def __init__(self):
        super(Conv_net, self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            #nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            #nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            #nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            #nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),

            nn.Flatten(),

            nn.Linear(256*2*2, 64*2*2),
            nn.Linear(64*2*2, 32*2*2),
            nn.Linear(32*2*2, num_class)
        )

    def forward(self, x):
        x = self.net(x)
        return x

convolution = Conv_net().to(device)




def losses(pred, correct):
    get_loss = nn.CrossEntropyLoss()
    return get_loss(pred, correct)

optimizer = torch.optim.Adam(convolution.parameters(), lr=learning_rate, weight_decay=0.00001)

for i in range(epoch):
    loss_fin = 0
    accuracy = 0
    for data in train_data:
        image, label = data
        #print(np.shape(image))
        #print(np.shape(label))
        pred = convolution(image)
        optimizer.zero_grad()
        loss = losses(pred, label)
        pred_label = torch.argmax(pred, dim=1)
        sub_acc = torch.sum(pred_label == label).item()
        accuracy += sub_acc / len(label)
        loss.backward()
        optimizer.step()
        loss_fin += loss.item()
    accuracy /= len(train_data)
    num_correct = 0
    for test_data in test_set:
        image_t, label_t = test_data
        pred_t = convolution(image_t[None, :, :, :])
        pred_t_label = torch.argmax(pred_t)
        if (pred_t_label == label_t):
            num_correct += 1
    test_accuracy = num_correct / len(test_set)
    loss_of_epoch = loss_fin / len(train_data)

    print("Epoch: %s" % (i))
    print("loss: %s" % (loss_of_epoch))
    print("Training_accuracy: %s" % accuracy)
    print("test accuracy: %s" % test_accuracy)