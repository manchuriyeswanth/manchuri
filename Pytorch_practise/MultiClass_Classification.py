import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Multiclass Classification")

NUM_CLASS = 4
NUM_FEATURE = 2
NUM_SAMPLES = 1000

X,y = make_blobs(NUM_SAMPLES,NUM_FEATURE,cluster_std=1.5,random_state=42)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)

print(X[:5],y[:5])

X_train, X_test , Y_train, Y_test = train_test_split(X,
                                                     y,
                                                     test_size=0.2,
                                                     random_state=42)

plt.figure()
plt.scatter(X_train[:,0],X_train[:,1], c=Y_train, cmap = plt.cm.RdYlBu)
plt.show()

## 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

## 
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self,In,Hi,Out):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=In, out_features=Hi),
            nn.ReLU(),
            nn.Linear(in_features=Hi,out_features=Hi),
            nn.ReLU(),
            nn.Linear(in_features=Hi, out_features=Out)
        )
    def forward(self,X):
        return self.linear_stack(X)


blob_model = MyModel(In=NUM_FEATURE,Hi=10,Out=NUM_CLASS).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=blob_model.parameters(),lr=0.1)


torch.manual_seed(42)
epochs = 100

X_blob_train, y_blob_train = X_train.to(device), Y_train.to(device)
X_blob_test, y_blob_test = X_test.to(device), Y_test.to(device)

def acc(y_true,y_pred):
    return (torch.eq(y_true,y_pred).sum().item()/len(y_true))*100
with mlflow.start_run():
        mlflow.set_tag("developer","yeswanth")
        for epoch in range(epochs):
            blob_model.train()

            y_logits = blob_model(X_blob_train)
            y_preds = torch.softmax(y_logits,dim=1).argmax(dim=1)

            train_loss = loss_fn(y_logits, y_blob_train)
            train_acc = acc(y_blob_train,y_preds)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            blob_model.eval()
            with torch.no_grad():
                test_logits = blob_model(X_blob_test)
                test_preds = torch.softmax(test_logits,dim=1).argmax(dim=1)

                test_loss = loss_fn(test_logits,y_blob_test)
                test_acc = acc(y_blob_test,test_preds)
            mlflow.log_metric("lr","0.01")
            mlflow.log_metric("Train_loss",train_loss,epoch)
            mlflow.log_metric("Test_loss",test_loss,epoch)
            if epoch%10 == 0:
                print(f"Epoch={epoch}, Train_loss :{train_loss:.2f}, Train_acc:{train_acc:.2f}, Test_loss:{test_loss:.2f}, Test_acc:{test_acc:.2f}")

    

import pickle
with open('model.pkl','wb') as f:
     pickle.dump(blob_model,f)

file = open('model.pkl','rb')
mymodel = pickle.load(file=file)
print(mymodel.state_dict())
