import numpy as np
import wandb
from sklearn.datasets import load_breast_cancer
from torch import nn
import torch
from sklearn.model_selection import train_test_split

# Start a new wandb run to track this script.
run = wandb.init(
    name="Try 3 layers with both tanh",
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="maxpendse-projects",
    # Set the wandb project where this run will be logged.
    project="first-nn-breast-cancer",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.01,
        "epochs": 25,
        "dataset": "Breast Cancer SKLearn",
    },
)

# Choose the device needed
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Load the data you need
data = load_breast_cancer()
epochs = run.config.epochs

X_train, X_test, Y_train, Y_test = train_test_split(
    data["data"],
    data["target"],
    test_size=0.15,
    random_state=42,
    shuffle=True,
)


# Normalize the data
xmean = X_train.mean()
std = X_train.std()

X_train = (X_train - xmean) / (std + 1e-8)  # eps prevents div-by-zero
X_test  = (X_test - xmean) / (std + 1e-8)


X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1).to(device)

print(X_train)

# Initialize the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(30, 45)
        self.act1 = nn.Tanh()
        self.lienar2 = nn.Linear(45,20)
        self.act2 = nn.Tanh()
        self.linear3 = nn.Linear(20,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits_1 = self.linear1(x)
        act_1 = self.act1(logits_1)
        logits_2 = self.lienar2(act_1)
        relu_2 = self.act2(logits_2)
        logits_3 = self.linear3(relu_2)
        normalized_output = self.sigmoid(logits_3)
        return normalized_output

# Initialize the model on the device
model = NeuralNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=run.config.learning_rate)
loss_fn = nn.MSELoss()
epoch_loss = 0

# Train linear regression with gradient descent.
model.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    model_outputs = model(X_train)
    loss = loss_fn(model_outputs, Y_train)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        model_test_outputs = model(X_test)
        preds = (model_test_outputs > 0.5).float()
        accuracy = (preds == Y_test).float().mean()
    
    epoch_loss += loss.item()
    # loss_test = loss_fn(model_test, Y_test)
    run.log({"loss_train": epoch_loss/(epoch + 1), "test_accuracy": accuracy})
    # run.log({"loss_train": loss.item(), "loss_test": loss_test.item()})

with torch.no_grad():
    model_test = model(X_test)

    table = wandb.Table(columns=["predicted", "actual"])
    for pred, actual in zip(model_test.cpu().numpy(), Y_test.cpu().numpy()):
        table.add_data(int(pred[0]), int(actual[0]))
    
    run.log({"predictions_vs_actual": table})

# Finish the run and upload any remaining data.
run.finish()
