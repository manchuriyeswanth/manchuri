from sklearn.datasets import make_circles

n_samples = 1000

X,y = make_circles(n_samples=n_samples,noise =0.01, random_state=42)

print(X[:5])
print(y[:5])

import pandas as pd
circles = pd.DataFrame({"X1":X[:,0],
                        "X2":X[:,1],
                        'label':y})
print(circles.head(10))
print(circles.label.value_counts())
import matplotlib.pyplot as plt
plt.scatter(circles.X1,circles.X2,c=y,cmap=plt.cm.RdYlBu)
plt.show()
## Always print shapes to avoid mismatches 
print(X.shape,y.shape)

## Convert numpy to torch tensor
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X[:5],y[:5])

## 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print(len(X_train),len(y_train),len(X_test),len(y_test))

## 
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print(X_test.shape)
## 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
    
    def forward(self,X):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(X)))))

model = Model().to(device)
print(model)

## 
untrained_preds = model(X_test.to(device))
print(untrained_preds[:10])
print(y_test[:10])

## 

loss_fn = nn.BCEWithLogitsLoss()

optim = torch.optim.SGD(params=model.parameters(),lr=0.1)

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = correct/len(y_true)*100
    return acc


## 

torch.manual_seed(42)
epochs = 1000

X_train, y_train = (X_train.to(device)), (y_train.to(device))
X_test , y_test = (X_test.to(device)), (y_test.to(device))

for epoch in range(epochs):

    model.train()

    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,y_train)

    acc = accuracy_fn(y_true=y_train,y_pred=y_pred)

    optim.zero_grad()
    loss.backward()
    optim.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        y_logits = torch.round(torch.sigmoid(y_pred))
        test_loss = loss_fn(y_logits,y_test)
        test_acc = accuracy_fn(y_test,y_logits)
    
    if (epoch%10 == 0):
        print(f"Epoch:{epoch}, Train Loss:{loss:.2f}, Accuracy:{acc:.2f}, Test_loss = {test_loss:.2f}, Test_Accuracy={test_acc:.2f}")


import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py",'wb') as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

def decision_boundary(model,X,y):
    plt.figure(figsize=[12,6])
    plt.subplot(1,2,1)
    plt.title('Train')
    plot_decision_boundary(model,X,y)
    plt.subplot(1,2,2)
    plt.title('Test')
    plot_decision_boundary(model,X,y)
    plt.show()

decision_boundary(model,X_train,y_train)
decision_boundary(model,X_test,y_test)
          