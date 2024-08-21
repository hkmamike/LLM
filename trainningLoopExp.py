import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
                
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
 
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
 
    def __len__(self):
        return self.labels.shape[0]
 
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
train_ds = ToyDataset(X_train, y_train)
 
torch.manual_seed(123)
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 3
 
for epoch in range(num_epochs): 
    
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
 
        logits = model(features)
        
        loss = F.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")
 
model.eval()
with torch.no_grad():
    outputs = model(X_train)

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
predictions = torch.argmax(outputs, dim=1)
print(probas)
print(predictions)
print(torch.sum(predictions == y_train))


def compute_accuracy(model, dataloader):
 
    model = model.eval()
    correct = 0.0
    total_examples = 0
    
    for idx, (features, labels) in enumerate(dataloader):
        
        with torch.no_grad():
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
 
    return (correct / total_examples).item()

print(compute_accuracy(model, train_loader))