import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Rastrigin function
import BP # loading Buggy-pinball algorithm
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
    # Forward pass
    predictions = model(X)

    # Compute and print loss
    loss = criterion(predictions, 0)  # target =0
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
 
    optimizer.zero_grad()  # Zero gradients 
    loss.backward() #  backward pass
    optimizer.step() #  update weights

print("Optimized Weight:", model.weight.data)

'''
------OUTPUT------
Epoch 1/100, Loss: 3.654061583802104e-05
Epoch 2/100, Loss: 3.0527993658324704e-05
Epoch 3/100, Loss: 2.550472527218517e-05
Epoch 4/100, Loss: 2.1308018403942697e-05
Epoch 5/100, Loss: 1.780185993993655e-05
Epoch 6/100, Loss: 1.4872632164042443e-05
Epoch 7/100, Loss: 1.242539565282641e-05
Epoch 8/100, Loss: 1.0380844287283253e-05
Epoch 9/100, Loss: 8.672715921420604e-06
Epoch 10/100, Loss: 7.24565234122565e-06
Epoch 11/100, Loss: 6.053407105355291e-06
Epoch 12/100, Loss: 5.057341695646755e-06
Epoch 13/100, Loss: 4.225175416650018e-06
Epoch 14/100, Loss: 3.5299381124787033e-06
Epoch 15/100, Loss: 2.949099780380493e-06
Epoch 16/100, Loss: 2.4638366085127927e-06
Epoch 17/100, Loss: 2.058421159745194e-06
Epoch 18/100, Loss: 1.7197158967974246e-06
Epoch 19/100, Loss: 1.4367431049322477e-06
Epoch 20/100, Loss: 1.2003324627585243e-06
Epoch 21/100, Loss: 1.0028220458480064e-06
Epoch 22/100, Loss: 8.378115694540611e-07
Epoch 23/100, Loss: 6.999526931394939e-07
Epoch 24/100, Loss: 5.84778092616034e-07
Epoch 25/100, Loss: 4.885550310973485e-07
Epoch 26/100, Loss: 4.081651354681526e-07
Epoch 27/100, Loss: 3.4100310131179867e-07
Epoch 28/100, Loss: 2.848923088549782e-07
Epoch 29/100, Loss: 2.3801432291747915e-07
Epoch 30/100, Loss: 1.9884998891939176e-07
Epoch 31/100, Loss: 1.6612995068499004e-07
Epoch 32/100, Loss: 1.3879387950055389e-07
Epoch 33/100, Loss: 1.1595586357771026e-07
Epoch 34/100, Loss: 9.687577318118201e-08
Epoch 35/100, Loss: 8.093521586260977e-08
Epoch 36/100, Loss: 6.761761994766857e-08
Epoch 37/100, Loss: 5.649139112051671e-08
Epoch 38/100, Loss: 4.719594315361064e-08
Epoch 39/100, Loss: 3.943003079598384e-08
Epoch 40/100, Loss: 3.294196204706168e-08
Epoch 41/100, Loss: 2.752148553497591e-08
Epoch 42/100, Loss: 2.2992926673737202e-08
Epoch 43/100, Loss: 1.9209524637631148e-08
Epoch 44/100, Loss: 1.6048668172174985e-08
Epoch 45/100, Loss: 1.3407920995689437e-08
Epoch 46/100, Loss: 1.1201697347473782e-08
Epoch 47/100, Loss: 9.35849975292058e-09
Epoch 48/100, Loss: 7.818591107877637e-09
Epoch 49/100, Loss: 6.532072216458573e-09
Epoch 50/100, Loss: 5.457243990036886e-09
Epoch 51/100, Loss: 4.559274735527197e-09
Epoch 52/100, Loss: 3.80906328700803e-09
Epoch 53/100, Loss: 3.1822957602400948e-09
Epoch 54/100, Loss: 2.658661069077084e-09
Epoch 55/100, Loss: 2.2211881223199725e-09
Epoch 56/100, Loss: 1.855700038078112e-09
Epoch 57/100, Loss: 1.5503516248571714e-09
Epoch 58/100, Loss: 1.295247020749457e-09
Epoch 59/100, Loss: 1.0821190610244003e-09
Epoch 60/100, Loss: 9.040604376231443e-10
Epoch 61/100, Loss: 7.553007108640486e-10
Epoch 62/100, Loss: 6.31018848284981e-10
Epoch 63/100, Loss: 5.271870162637526e-10
Epoch 64/100, Loss: 4.404403786129052e-10
Epoch 65/100, Loss: 3.67967545322756e-10
Epoch 66/100, Loss: 3.0741983958471053e-10
Epoch 67/100, Loss: 2.568350809806219e-10
Epoch 68/100, Loss: 2.1457383092560178e-10
Epoch 69/100, Loss: 1.7926647666310913e-10
Epoch 70/100, Loss: 1.4976885009954088e-10
Epoch 71/100, Loss: 1.2512493818839232e-10
Epoch 72/100, Loss: 1.0453610893579324e-10
Epoch 73/100, Loss: 8.733507667058049e-11
Epoch 74/100, Loss: 7.296442616766541e-11
Epoch 75/100, Loss: 6.095839805153602e-11
Epoch 76/100, Loss: 5.092792160210635e-11
Epoch 77/100, Loss: 4.254792351776615e-11
Epoch 78/100, Loss: 3.554681896056344e-11
Epoch 79/100, Loss: 2.969772344707522e-11
Epoch 80/100, Loss: 2.4811075069464117e-11
Epoch 81/100, Loss: 2.0728503982714308e-11
Epoch 82/100, Loss: 1.7317704562036518e-11
Epoch 83/100, Loss: 1.4468140227541504e-11
Epoch 84/100, Loss: 1.2087462974985641e-11
Epoch 85/100, Loss: 1.0098514906242695e-11
Epoch 86/100, Loss: 8.436842370562747e-12
Epoch 87/100, Loss: 7.048591311314967e-12
Epoch 88/100, Loss: 5.888771675149895e-12
Epoch 89/100, Loss: 4.919795473790067e-12
Epoch 90/100, Loss: 4.110262043915958e-12
Epoch 91/100, Loss: 3.4339334761129825e-12
Epoch 92/100, Loss: 2.868892406623913e-12
Epoch 93/100, Loss: 2.3968273112767724e-12
Epoch 94/100, Loss: 2.0024383626943143e-12
Epoch 95/100, Loss: 1.6729446604066278e-12
Epoch 96/100, Loss: 1.397667788796686e-12
Epoch 97/100, Loss: 1.1676867155488435e-12
Epoch 98/100, Loss: 9.755480928283489e-13
Epoch 99/100, Loss: 8.150254017687264e-13
Epoch 100/100, Loss: 6.809159898250872e-13
Optimized Weight: tensor([ 3.6260e-07, -2.9485e-08])

'''


