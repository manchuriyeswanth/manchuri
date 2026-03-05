## Workflows Part 2
## y=mx+c 

## known params 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
weight = 0.8
bias = 0.3
x = torch.arange(0,1,0.02).unsqueeze(dim=1)
y = x * weight + bias
print(x[:10],y[:10])


##
train_split = int(0.8*len(x))
X_train, y_train = x[:train_split], y[:train_split]
X_test , y_test = x[train_split:], y[train_split:]
print(f"X_len:{len(x)}, X_train_len:{len(X_train)}")

## Visualizations

def plot_predictions(train_data=X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions = None):
    plt.figure(figsize = (10,7))
    plt.scatter(train_data,train_labels,c='b',s=4,label="Training Data")
    plt.scatter(test_data,test_labels,c='r',s=4,label="Testing Data")
    if predictions is not None:
        plt.scatter(test_data,predictions,c='g',s=4,label="Predictions")
    plt.legend(prop={"size":14})
    plt.show()

plot_predictions()

## Building Models

class LinearRegressionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                dtype=torch.float,
                                                requires_grad=True)) ## We need to update this variable by Gradient Descent
        self.bias = nn.Parameter(torch.randn(1,
                                             dtype = torch.float,
                                             requires_grad=True))
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.weights*x+self.bias 


torch.manual_seed(42)
model0 = LinearRegressionModule()
print(model0.state_dict())

with torch.inference_mode():
    y_preds = model0(X_test)

print(f"Len X_test = {len(X_test)}")
plot_predictions(predictions=y_preds)

loss_fn = torch.nn.L1Loss()

optimizer = torch.optim.SGD(params=model0.parameters(), 
                            lr=0.01) # always the optimizer takes the parameters that we need to optimize on

torch.manual_seed(42)

epochs=100
train_loss_values=[]
test_loss_values=[]
epoch_count=[]
for epoch in range(epochs):
    model0.train()
    y_pred = model0(X_train)
    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model0.eval()
    with torch.no_grad():
        y_pred = model0(X_test)
        test_loss = loss_fn(y_pred,y_test.type(torch.float))

        if epoch % 10 ==0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch:{epoch}, TrainingLoss: {loss}, Testloss:{test_loss}")

plt.plot(epoch_count,train_loss_values,label="Train Loss")
plt.plot(epoch_count,test_loss_values,label = "Test Loss")
plt.title("Training & Test Loss")
plt.xlabel("Epoch Count")
plt.ylabel("Loss")
plt.legend()
plt.show()

## Model Parameters Updated
print("Latest Model Params")
print(model0.state_dict())

## Save and Load and test 
from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)
MODEL_NAME="MYLRE.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
model_saved = torch.save(obj=model0.state_dict(),f=MODEL_SAVE_PATH)

loaded_model = LinearRegressionModule()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print(loaded_model.state_dict())
loaded_model.eval()
with torch.no_grad():
    loaded_model_preds = loaded_model(X_test)

print((y_pred == loaded_model_preds).sum(), len(y_pred))