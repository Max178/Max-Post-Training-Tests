import numpy as np
import wandb
from sklearn.datasets import load_breast_cancer
from torch import nn
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc

# Start a new wandb run to track this script.
run = wandb.init(
    name="Add extra metrics",
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="maxpendse-projects",
    # Set the wandb project where this run will be logged.
    project="Max-First-Project",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "epochs": 25,
        "dataset": "Breast Cancer SKLearn",
        "batch_size": 50,
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
    test_size=0.3,
    random_state=42,
    shuffle=True,
)


# Normalize the data
xmean = X_train.mean(axis = 0)
std = X_train.std(axis = 0)
X_train = (X_train - xmean) / (std + 1e-8)  # eps prevents div-by-zero
X_test  = (X_test - xmean) / (std + 1e-8)

# Convert to Tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1).to(device)

dataset = TensorDataset(X_train, Y_train)
data_loader = DataLoader(dataset, batch_size=run.config.batch_size, shuffle=True)

# Initialize the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(30, 45)
        self.act1 = nn.Tanh()
        self.linear2 = nn.Linear(45,20)
        self.act2 = nn.Tanh()
        self.linear3 = nn.Linear(20,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits_1 = self.linear1(x)
        act_1 = self.act1(logits_1)
        logits_2 = self.linear2(act_1)
        relu_2 = self.act2(logits_2)
        logits_3 = self.linear3(relu_2)
        normalized_output = self.sigmoid(logits_3)
        return normalized_output

# Initialize the model on the device
model = NeuralNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=run.config.learning_rate)
loss_fn = nn.BCELoss()
epoch_loss = 0

# Train linear regression with gradient descent.
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for batch_x, batch_y in data_loader:
        optimizer.zero_grad()
        model_outputs = model(batch_x)
        loss = loss_fn(model_outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss * len(batch_x)

    with torch.no_grad():
        model_test_outputs = model(X_test)
        preds = (model_test_outputs > 0.5).float()
        accuracy = (preds == Y_test).float().mean()
    
    running_loss = running_loss/len(X_train)
    epoch_loss += running_loss
    run.log({"loss_train": epoch_loss/(epoch + 1), "test_accuracy": accuracy})


with torch.no_grad():
    model_test = model(X_test)

    table = wandb.Table(columns=["predicted", "actual"])
    for pred, actual in zip(model_test.cpu().numpy(), Y_test.cpu().numpy()):
        table.add_data((pred[0] > 0.5), int(actual[0]))
    
    run.log({"predictions_vs_actual": table})

    y_true = Y_test.cpu().numpy().flatten()
    y_pred_prob = model_test.cpu().numpy().flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_prob)

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall_curve, precision_curve)

    run.log({"accuracy": accuracy, "precision": precision, "recall": recall, "auc_roc": auc_roc, "auc_pr": auc_pr})

# Finish the run and upload any remaining data.
run.finish()
