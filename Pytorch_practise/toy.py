import torch

## Scalar
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)
print(scalar.item())

## Vector
vector = torch.tensor([7,7])
print(vector)
print(vector.ndim)

## Matrix
matrix = torch.tensor([[7,7],[6,7]])
print(matrix)
print(matrix.ndim)
print(matrix.shape)

##  Random

random_tensor = torch.rand(size=[3,4])
print(random_tensor)
print(random_tensor.shape,random_tensor.ndim)

## Zeroes

zeros = torch.zeros(size=[3,4])
print(zeros.shape)
print(zeros)

## Ones
ones = torch.ones(size=[3,4])
print(ones)
print(ones.shape,ones.ndim)
print(ones.dtype)

## Range

lst = torch.arange(0,10,1)
print(lst)

## Operations

tensor = torch.tensor([10,11,12])
print(tensor+10)
print(tensor*10)

## Matrix Multiplication "@" - Inner dimension must match

vec = torch.tensor([1,2,3])
print(vec@vec) # Matrix multiplicaton and added 

print(vec.matmul(vec))

import time
t1 = time.process_time()  ## process_time used to get time taken by a function and independent of System IO 
print(t1)
print(vec.matmul(vec))
t2 = time.process_time()
print((t2-t1)*1e6)

## torch dimension mismatches 
A = torch.rand(size = [3,4])
B = torch.rand(size = [3,4])
print(A@B.transpose(1,0))  # .transpose(dim0,dim1) shows the dimensions that has to be swapped for matrix multilpication to take place

## Linear operation or nn.linear module y=xA'+b or a simple matrix multiplication with weights given by linear layer
torch.manual_seed(42) ## denotes manual seed to have same output values every time
linear = torch.nn.Linear(in_features=4,out_features=6)
x = A
out  = linear(x)
print(x.shape)
print(out.shape)

## Reshape, stack, Squeeze, UnSqueeze

s = torch.arange(10,100,10)
print(s,s.shape)
print(s.ndim)
s_reshaped = s.reshape(1,9)
print(s_reshaped)
print(s_reshaped.ndim)
##
z_reshaped = s.view(1,9) 
z_reshaped[:,1]= 1
print(z_reshaped)
print(s)
z_concat = torch.concat((z_reshaped,z_reshaped), dim=1)
print(z_concat)
print(s_reshaped.shape)
##
s_squeeze = s_reshaped.squeeze()
print(s_squeeze.shape)


## s_unsqueeze

s_unsqueeze = s.unsqueeze(1)
print(s.shape,s_unsqueeze.shape)
## 
A = torch.rand(size =[224,224,3])
print(A.shape)
A_permuted = A.permute([2,0,1])
print(A_permuted.shape)
##
lst = torch.arange(1,10).reshape([1,3,3]) # think of 1 matrix of shape 3x3
print(lst,lst.ndim,lst.shape)
lst2 = torch.arange(1,10).reshape([3,3,1]) # 3 matrices of shape 3x1
print(lst2,lst2.ndim,lst2.shape)
lst3 = torch.arange(1,10).reshape([3,1,3]) # 3 matrices of shape 1x3
print(lst3,lst3.ndim,lst3.shape)

x= torch.tensor([[[1,2,3],[3,4,5],[6,7,8]]])
print(x)
print(x.shape)
print(x[:,0])
print(x[:,:,1])
## numpy to torch

import numpy as np
lst = np.arange(1,8)
print(lst)
t = torch.from_numpy(lst)
print(t)

## Torch Random numbers
seed = 42
torch.manual_seed(seed=seed)
A = torch.rand(size=[3,4])
print(A)

## Exercies
r = torch.rand(size=[7,7])
r2 = torch.rand(size=(1,7))
r3 = r@r2.transpose(0,1)
print(r3.shape,r3)
torch.manual_seed(0)
r = torch.rand(size=[7,7])
r2 = torch.rand(size=(1,7))
r3 = r@r2.transpose(0,1)
print(r3.shape,r3)
