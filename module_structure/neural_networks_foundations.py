import torch
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Inputs observerd

x = [-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5]

# Outputs observerd

y = [2.292810956,2.669656316,2.176859694,2.761872736,2.64112637,2.546430827,2.895934924,3.455978572,3.003456113,3.066457256,3.291001933,3.601666326,3.833641047,3.900596414,3.613502837,4.331765413,3.963925711,3.971332485,4.296742522,4.714925133,4.65354761,4.960615192,5.424617958,5.079345736,5.860704627,5.582586609,6.171985028,5.503825528,6.214933558,6.292708962,5.763454252,6.776152515,6.969193357,7.00409557,6.450674523,6.662626481,6.798758237,7.406862568,7.05617933,7.843912093]

# Drawing a plot with the data
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel('Inputs observed', fontsize=10)
ax.set_ylabel('Outputs observed', fontsize=10)
plot = ax.plot(x,y,'o')
plt.show()


def model(weight, x, bias):
    y_hat = weight * x + bias
    return y_hat


# Training a neural network will essentially involve changing the model for a slightly more elaborate one,
# with a few (or millions) more parameters


def loss_fn(y_hat, y):
    squared_diffs = (y_hat - y)**2
    mse = squared_diffs.mean()
    return mse


# Less loss is what we want - As less loss as possible

w = torch.ones(())
b = torch.zeros(())
x = torch.tensor(x)

y_hat = model(w, x, b)


def derivatedloss_fn(y_hat, y):
    dLoss_y_hat = 2 * (y_hat - y)
    return dLoss_y_hat


def derivatedmodel_dw(x, w, b):
    return x


def derivatedmodel_db(x, w, b):
    return 1.0


def grad_fn(x, y, y_hat, w, b):
    # Remember, you can use the chain rule to code the gradient
    dloss_dtp = derivatedloss_fn(y_hat, y)
    dloss_dw = dloss_dtp * derivatedmodel_dw(x, w, b)
    dloss_db = dloss_dtp * derivatedmodel_db(x, w, b)

    return torch.stack([dloss_dw.sum() / y_hat.size(0), dloss_db.sum() / y_hat.size(0)])


def training_loop(n_epochs, learning_rate, params, x, y, print_params=True):
    losses = []
    for epoch in range(1, n_epochs + 1):

        # Params we want to fit
        w, b = params

        # Setting the model we will use (a simple linear equation)
        y_hat = model(x, w, b)

        # Setting our gradient vector
        gradient_vector = grad_fn(x, y, y_hat, w, b)

        # Setting our Gradient Descent step
        params = params - learning_rate * gradient_vector

        ### Calculate the loss obtained ###
        loss = loss_fn(y_hat, y)

        # Saving the loss values in a list just to make a picture
        losses.append(loss)

        # Showing the loss obtained at different epochs
        if epoch in {1, 2, 3, 10, 11, 99, 100, 1000, 4000, 5000, 10000, 50000, 70000}:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            if print_params:
                print('    Params:', params)
                print('    Grad:  ', gradient_vector)
                print('    Loss:  ', loss)
        if epoch in {4, 12, 101}:
            print('...')

    return params, losses


x = torch.tensor(x)
y = torch.tensor(y)

n_epochs = 10000

params,losses = training_loop(
                n_epochs=n_epochs,
                learning_rate=1e-3,
                params=torch.rand(2),
                x=x,
                y=y)


print("\nThe model, after {0:d} epochs gets the following parameters: \n weight= {1:.2f} \n bias = {2:.2f}".format(n_epochs,
                                                                                                                   params[0].item(),
                                                                                                                   params[1].item()))

my_model = model(x, *params)

# Drawing a plot with the data
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel('Inputs observed', fontsize=10)
ax.set_ylabel('Outputs observed', fontsize=10)
ax.plot(x,y,'o')
ax.plot(x, my_model.detach())


# Drawing a plot with the losses
fig, ax = plt.subplots(figsize=(6, 4))
plt.xlabel("Epoch")
plt.ylabel("Computed Loss")
ax.plot(losses)


input_value = 20

test_val = torch.tensor([input_value])
prediction = model(test_val,*params)


print("The model predicts a value of {0:.2f} for the input {1:.2f}".format(float(input_value),prediction.item()))