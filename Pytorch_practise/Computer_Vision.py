import torch
import torch.nn as nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

print(f"{torchvision.__version__},{torch.__version__}")

## DataSet Preparation

train_data = datasets.FashionMNIST(root="data",
                                   train = True,
                                   transform = ToTensor(),
                                   target_transform=None,
                                   download=True)

test_data = datasets.FashionMNIST(root="data",
                                  train = False,
                                  transform = ToTensor(),
                                  target_transform=None,
                                  download=True)


image, label = train_data[0]
print(image, label)

print(len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets))

class_names = train_data.classes
print(class_names)

#plt.imshow(image.squeeze())
#plt.title(label)
#plt.show()

## 

BATCH_SIZE = 32

train_dataloader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)

test_dataloader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)

train_dataloader_batch, train_data_labels_batch = next(iter(train_dataloader))

print(train_dataloader_batch.shape, train_data_labels_batch.shape)

## Model 1 

class FashionMNIST_v1(nn.Module):
    def __init__(self,input,hidden, out):
        super().__init__()
        self.layer_stack=nn.Sequential(nn.Flatten(),
                                       nn.Linear(in_features=input, out_features=hidden),
                                       nn.Linear(in_features=hidden,out_features=out))
    def forward(self,X):
        return self.layer_stack(X)

torch.manual_seed(42)

model0 = FashionMNIST_v1(input=784,hidden=10,out=len(train_data.classes))

import requests
from pathlib import Path
if Path('helper_functions.py').is_file():
    print("helper_finctions already exists, skipping")
else:
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open('helper_functions.py','wb') as f:
        f.write(request.content)


from helper_functions import accuracy_fn
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=model0.parameters(),lr=0.1)

from timeit import default_timer as timer
def print_traintime(start:float,end:float,device:torch.device=None):
    total_time = end-start
    print(f"Totaltime taken to run on {device}:{total_time:.3f}sec")
    return total_time


from tqdm.auto import tqdm
torch.manual_seed(42)
time_on_start = timer()
epochs = 3

for epoch in range(epochs):
    print(f"Epoch-----:{epoch}")
    train_loss=0

    for batch,(X,y) in enumerate(train_dataloader):

        model0.train()

        y_pred= model0(X)
        loss = loss_fn(y_pred,y)
        train_loss = train_loss +loss
        optim.zero_grad()
        loss.backward()
        optim.step()


        if batch%400 ==0 :
            print(f" Looked at {batch}*{len(X)}/{len(train_dataloader.dataset)} samples")

    train_loss = train_loss/len(train_dataloader)

    test_loss,test_acc =0,0
    model0.eval()
    with torch.no_grad():
        for X,y in test_dataloader:
            test_pred = model0(X)
            test_loss += loss_fn(test_pred,y)

            test_acc +=accuracy_fn(y,test_pred.argmax(dim=1))
        test_loss = test_loss/len(test_dataloader)

        test_acc = test_acc/len(test_dataloader)
    print(f"Train_loss :{train_loss:.2f}, Test_loss :{test_loss:.2f}, Test_acc= {test_acc:.2f}")

end_time = timer()
print_traintime(time_on_start,end_time, device=str(next(model0.parameters()).device))


torch.manual_seed(42)

def eval_model(model:torch.nn.Module,
               data_loader : torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    
    loss, acc = 0,0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred,y)

            acc += accuracy_fn(y,y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {"model":model.__class__.__name__,
            "loss":loss.item(),
            "acc":acc}

model0_results = eval_model(model0,test_dataloader,loss_fn,accuracy_fn)
print("Model0 Results")
print(model0_results)

device = "cuda" if torch.cuda.is_available() else "cpu"

class FashionMNIST_v1(nn.Module):
    def __init__(self,inp,hid,out):
        super().__init__()
        self.stack = nn.Sequential(nn.Flatten(),
                                   nn.Linear(inp,hid),
                                   nn.ReLU(),
                                   nn.Linear(hid,out),
                                   nn.ReLU())
    def forward(self,X):
        return self.stack(X)

torch.manual_seed(42)

model1 = FashionMNIST_v1(inp=784,
                         hid=10,
                         out=len(class_names)).to(device)

print(next(model1.parameters()).device)

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
train_time_start_on_gpu = timer()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=model1.parameters(),lr=0.1)
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model1, 
        loss_fn=loss_fn,
        optimizer=optim,
        accuracy_fn=accuracy_fn
    )
    test_step(data_loader=test_dataloader,
        model=model1,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_traintime(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)

torch.manual_seed(42)

# Note: This will error due to `eval_model()` not using device agnostic code 
model_1_results = eval_model(model=model1, 
    data_loader=test_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn) 
print("Model1 results")
print(model_1_results)


## CNN Module

class FashionMNIST_2(nn.Module):
    def __init__(self,inp,hid,out):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(inp,hid,3,1,1),
                                    nn.ReLU(),
                                    nn.Conv2d(hid,hid,3,1,1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2))
        self.layer2 = nn.Sequential(nn.Conv2d(hid,hid,3,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(hid,hid,3,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(hid*7*7,out))
    
    def forward(self,X):
        return self.classifier(self.layer2(self.layer1(X)))

torch.manual_seed(42)
model2 = FashionMNIST_2(inp=1,hid=10,out=len(class_names)).to(device)

print(model2)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=model2.parameters(),lr=0.1)

torch.manual_seed(42)

from timeit import default_timer as timer
epochs = 3
start_time = timer()
for epoch in tqdm(range(epochs)):
    print(f"------------Epoch:{epoch}-------------")
    train_step(model=model2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optim,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(data_loader=test_dataloader,
              model=model2,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
end_time= timer()
print_traintime(start_time,end_time,device=device)
print("Model2 Results")
model2_results = eval_model(model=model2,
                            data_loader=test_dataloader,
                            loss_fn=loss_fn,
                            accuracy_fn=accuracy_fn)
import pandas as pd

results = pd.DataFrame([model0_results, model_1_results, model2_results])


print(results)

def make_prediction(model:torch.nn.Module,
                    data:list,
                    device:torch.device):
    pred_probs=[]
    model.eval()
    with torch.no_grad():
        for sample in data:
            sample = torch.unsqueeze(sample,dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(),dim=0)
            pred_probs.append(pred_prob)
    return torch.stack(pred_probs) # List into tensor

import random
random.seed(42)
test_samples=[]
test_labels=[]
for sample,label in random.sample(list(test_data),k=9):
    test_samples.append(sample)
    test_labels.append(label)


pred_probs = make_prediction(model2,test_samples,device)
pred_classes = pred_probs.argmax(dim=1)
print(test_labels,pred_classes)

## Plot predictions

plt.figure(figsize=[9,9])
nrow=3
ncol=3
for i,sample in enumerate(test_samples):
    plt.subplot(nrow,ncol,i+1)
    plt.imshow(sample.squeeze(dim=0),cmap='gray')
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]
    title_text = f"Pred={pred_label}, truth_label={truth_label}"
    if pred_label==truth_label:
        plt.title(title_text,fontsize=10,c='g')
    else:
        plt.title(title_text,fontsize=10,c='r')
    plt.axis(False)

# Import tqdm for progress bar
from tqdm.auto import tqdm

# 1. Make predictions with trained model
y_preds = []
model2.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc="Making predictions"):
    # Send data and targets to target device
    X, y = X.to(device), y.to(device)
    # Do the forward pass
    y_logit = model2(X)
    # Turn predictions from logits -> prediction probabilities -> predictions labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
    # Put predictions on CPU for evaluation
    y_preds.append(y_pred.cpu())
# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)

import mlxtend

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
);


from pathlib import Path 
MODELPATH = Path("models")
MODELPATH.mkdir(parents=True,exist_ok=True)

MODELNAME = "Computer_Vision_practise.pth"
MODELSAVE_PATH = MODELPATH / MODELNAME

print(f"Saving model to :{MODELSAVE_PATH}")

torch.save(obj=model2.state_dict(),
           f=MODELSAVE_PATH)


loaded_model2 = FashionMNIST_2(inp=1,
                               hid=10,
                               out=10)

loaded_model2.load_state_dict(torch.load(f=MODELSAVE_PATH))

loaded_model2= loaded_model2.to(device)

torch.manual_seed(42)

loaded_model2_results = eval_model(model = loaded_model2,
                                   data_loader=test_dataloader,
                                   loss_fn=loss_fn,
                                   accuracy_fn=accuracy_fn)

print(loaded_model2_results)


torch.isclose(torch.tensor(model2_results["loss"]), 
              torch.tensor(loaded_model2_results["loss"]),
              atol=1e-08, # absolute tolerance
              rtol=0.0001) # relative tolerance


