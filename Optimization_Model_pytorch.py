import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Rastrigin function
import BP_pytorch
import functions
DIMENSIONS = 2  # without y axis
# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.weight =nn.Parameter(torch.zeros(DIMENSIONS))

    def forward(self, x):
        return self.weight * x

# BP implementation where i am calling main Buggy-pinball --this is just a trial
def buggy_pinball():

    # define hyper-parameters
    START_RESULTANT = -.5  # start step size
    END_RESULTANT = .00001  # end step size
    START_ANGLE = 30
    END_ANGLE = 60
    MAX_ITER = 1000
    NUM_OF_STEPS = 15
    fitness = functions.rastrigin2
    REFINEMENT_PREC = 0.0000001
    low = -5.12
    up = 5.12

    global_best_position = BP_pytorch.main(DIMENSIONS, low, up, fitness,NUM_OF_STEPS, START_RESULTANT , END_RESULTANT ,START_ANGLE, END_ANGLE, MAX_ITER)
    return global_best_position

# Create a linear regression model
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # You can choose an appropriate learning rate

# Run PSO to optimize the Rastrigin function
best_position = buggy_pinball()

# Set the model's weight to the optimized value
model.weight.data = torch.tensor(best_position, dtype=torch.float32)

# Generate some random input data for prediction
X = torch.tensor(np.random.uniform(-5.12, 5.12, size=(1000, 1)), dtype=torch.float32)

# Define the number of epochs and target
num_epochs = 100
target = torch.tensor(0.0)
# Training loop
for epoch in range(num_epochs):
    # Forward pass: Compute predicted y by passing x to the model
    predictions = model(X)

    # Compute and print loss
    loss = criterion(predictions, target)  # You need to define the target variable
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Zero gradients, perform backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Optimized Weight:", model.weight.data)
'''
# Set the optimal weights to the model
with torch.no_grad():
    model.layer1.weight = nn.Parameter(best_position.unsqueeze(0))

# Test the optimized model
test_input = torch.tensor(best_position, dtype=torch.float32)
predicted_output = model(test_input)
print(f"Predicted output for optimized solution: {predicted_output.item()}")
'''




