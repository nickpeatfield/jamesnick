import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from torch import nn, optim


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

def make_acc_dataset(model, dataset):
  n = len(dataset)
  for i in range(n):
      x, y = dataset[i][0], dataset[i][1]
      with torch.no_grad():
          pred = model(x)
      if (not(pred[0].argmax(0) - y)):
          dataset.targets[i] = 1
      else:
          dataset.targets[i] = 0
  return dataset

#%%

# Get cpu or gpu device for training.
device = "cpu"
print("Using {} device".format(device))
# Define model
class NeuralNetworkClass(nn.Module):
    def __init__(self):
        super(NeuralNetworkClass, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x).to(device)
        return logits

class_model = NeuralNetworkClass().to(device)
print(class_model)

class NeuralNetworkPred(nn.Module):
    def __init__(self):
        super(NeuralNetworkPred, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x).to(device)
        return logits


pred_model = NeuralNetworkPred().to(device)
print(pred_model)

loss_fn_class = nn.CrossEntropyLoss()
optimizer_class = torch.optim.SGD(class_model.parameters(), lr=1e-3)
loss_fn_pred = nn.CrossEntropyLoss()
optimizer_pred = torch.optim.SGD(pred_model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn, min_loss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if test_loss < min_loss:
        print(f"Early Stopping: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return 1
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 0

def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


torch.save(class_model.state_dict(), f"blankallclass_model.pth")
torch.save(pred_model.state_dict(), f"blankpred_model.pth")

epochs = 50
model_gens = 10
for g in range(model_gens):
    # Reset all the models
    # Full 10 class
    print(f"New Gen Time \n")
    class_model = NeuralNetworkClass()
    class_model.load_state_dict(torch.load("blankallclass_model.pth"))
    loss_fn_class = nn.CrossEntropyLoss()
    optimizer_class = torch.optim.SGD(class_model.parameters(), lr=1e-3)
    # Just predict
    pred_model = NeuralNetworkPred()
    pred_model.load_state_dict(torch.load("blankpred_model.pth"))
    loss_fn_pred = nn.CrossEntropyLoss()
    optimizer_pred = torch.optim.SGD(pred_model.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"Epoch {t+1} of gen {g+1}\n-------------------------------")
        train(train_dataloader, class_model, loss_fn_class, optimizer_class)
        early_stopping = test(test_dataloader, class_model, loss_fn_class, 1)
        if early_stopping == 1 or t == epochs:
            torch.save(class_model.state_dict(), f"{g+1}_allclass_model.pth")
            print(f"Saved PyTorch Model State to {g+1}_allclass_model.pth")
            print(f"Predicting {g+1}\n-------------------------------")
            pred_train_dataset = make_acc_dataset(class_model,training_data)
            pred_test_dataset = make_acc_dataset(class_model,test_data)
            pred_train_dataloader = DataLoader(pred_train_dataset, batch_size=batch_size)
            pred_test_dataloader = DataLoader(pred_test_dataset, batch_size=batch_size)
            for p in range(epochs):
                print(f"Epoch {p+1} of gen {g+1} pred class \n-------------------------------")
                train(pred_train_dataloader, pred_model, loss_fn_pred, optimizer_pred)
                early_stopping = test(pred_test_dataloader, pred_model, loss_fn_pred, 0.5)
                if early_stopping == 1 or p == epochs:
                    torch.save(pred_model.state_dict(), f"{g+1}_pred_model.pth")
                    print(f"Saved PyTorch Model State to {g+1}_pred_model.pth")
                    break
            break


print("Done!")



