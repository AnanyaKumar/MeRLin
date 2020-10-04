import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import  torchvision.datasets as datasets 
import  torchvision.models as models 
import  torchvision.transforms as transforms
import  nvidia.dali.ops as ops
import  nvidia.dali.types as types
from    nvidia.dali.pipeline import Pipeline
from    nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
import  os

def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp