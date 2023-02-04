#import libraries
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd

#convert xls to csv file
from matplotlib import pyplot as plt

#data_xls = pd.read_excel('filename', index_col=0)
#data_xls.to_csv('filename', encoding='utf-8')

dataset = pd.read_csv('filename')
#print(dataset)
#dataset operations (drop unecessary columns)
#dataset = dataset.drop(dataset.columns[[0, 1]], axis=1)

#convert string to numericial values
#dataset.at[dataset['Label'] == 'Genuine', ['Label']] = 0
#dataset.at[dataset['Label'] == 'Posed', ['Label']] = 1

dataset = dataset.apply(pd.to_numeric)

#split training and testing datasets
n = np.random.rand(len(dataset)) < 0.8
data_train = dataset[n]
#print(data_train)
data_test = dataset[~n]
#print(data_test)

#train dataset
data_train_array = data_train.values

#the last column is target
#split features and targets
#features in the training dataset
x_train_array = data_train_array[:, :6]
#print(x_train_array)
#targets in the training dataset
y_train_array = data_train_array[:, 6]
#print(y_train_array)

X_train = torch.tensor(x_train_array, dtype=torch.float)
Y_train = torch.tensor(y_train_array, dtype=torch.long)

#input layers: 6 neurons (features of anger)
#output layers: 2 neurons (number of labels)
#hidden layers: 10
#define parameters in the neural network
input_size = 6
output_size = 2
hidden_size = 10
learning_rate = 0.02
num_epoch = 500

# define a customised neural network structure
class TwoLayerNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(TwoLayerNet, self).__init__()
        # define linear hidden layer output
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        # define linear output layer output
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        """
            In the forward function we define the process of performing
            forward pass, that is to accept a Variable of input
            data, x, and return a Variable of output data, y_pred.
        """
        # get hidden layer input
        h_input = self.hidden(x)
        # define activation function for hidden layer
        h_output = torch.sigmoid(h_input)
        # get output layer output
        y_pred = self.out(h_output)

        return y_pred

# define a neural network using the customised structure
net = TwoLayerNet(input_size, hidden_size, output_size)

# define loss function
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []

# train a neural network
for epoch in range(num_epoch):
    # Perform forward pass: compute predicted y by passing x to the model.
    Y_pred = net(X_train)

    # Compute loss
    loss1 = loss_func(Y_pred, Y_train)
    all_losses.append(loss1.item())

    # print progress
    if epoch % 50 == 0:
        # convert three-column predicted Y values to one column for comparison
        _, predicted = torch.max(Y_pred, 1)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y_train.data.numpy()

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epoch, loss1.item(), 100 * sum(correct)/total))

    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss1.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()

    plt.figure()
    plt.plot(all_losses)
    plt.show()

#test dataset
data_test_array = data_test.values

#the last column is target
#split features and targets
x_test = data_test_array[:, :6]
#print(x_test_array)
y_test = data_test_array[:, 6]
#print(y_test_array)

X_test = torch.tensor(x_test, dtype=torch.float)
Y_test = torch.tensor(y_test, dtype=torch.long)

Y_pred_test = net(X_test)

# get prediction
# convert three-column predicted Y values to one column for comparison
_, predicted_test = torch.max(Y_pred_test, 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

loss2 = loss_func(Y_pred_test, Y_test)

print('Testing Loss: %.4f' % (loss2.item()))
print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
