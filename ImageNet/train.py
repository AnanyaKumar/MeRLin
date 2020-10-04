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
from    networks import *
from    dali_data import *
from    utils import * 
from    merlin import * 

train_loader = get_imagenet_iter_dali(type='train', image_dir='/data/ImageNet2012/train/', batch_size=128,
                                          num_threads=4, crop=224, device_id=0, num_gpus=4)
train_loader1 = get_imagenet_iter_dali(type='train', image_dir='/data/finetune/stanford_cars/train/', batch_size=512,
                                          num_threads=4, crop=224, device_id=0, num_gpus=4)

feature_extractor = ResNetFc(model_name='resnet18',model_path='/data/pytorchModels/resnet18.pth').cuda()
cls = CLS(feature_extractor.output_num(),1000, bottle_neck_dim=False).cuda()
feature_extractor = nn.DataParallel(feature_extractor)
cls = nn.DataParallel(cls)

criterion = torch.nn.CrossEntropyLoss()
criterion1 = torch.nn.MSELoss()

optimizer = torch.optim.SGD(feature_extractor.parameters(),lr=1e-2,weight_decay=5e-4,momentum=0.9)
optimizer1 = torch.optim.SGD(cls.parameters(),lr=1e-2,weight_decay=5e-4,momentum=0.9)

feature_extractor.train()
cls.train()
step = 0

for epoch in range(19200):
    if epoch % 100 == 1:
        torch.save(feature_extractor.state_dict(),'checkpoint/'+str(epoch) +'fe' +'.pth')
        torch.save(cls.state_dict(),'checkpoint/'+str(epoch) +'cls' +'.pth')
    for i,(data,data1) in enumerate(zip(train_loader,train_loader1)):
        
        step = step + 1
        if epoch > 2400 & epoch <= 6400:
            for g in optimizer.param_groups:
                g['lr'] = 1e-3
            for g in optimizer1.param_groups:
                g['lr'] = 1e-3
        if epoch > 6400:
            for g in optimizer.param_groups:
                g['lr'] = 1e-4
            for g in optimizer1.param_groups:
                g['lr'] = 1e-4
        image = data[0]["data"].cuda(non_blocking=True)
        target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        image1 = data1[0]["data"].cuda(non_blocking=True)
        target1 = data1[0]["label"].squeeze().long().cuda(non_blocking=True)
        
        fss,fs = cls(feature_extractor(image))
        ftt = feature_extractor(image1)
        
        target_label1, target_test_pred = merlin_linear(fss,ftt,target1,256, 256, 200)
        
        ce1 = criterion(target_test_pred, target_label1)
        ce = criterion(fs,target)
        
        loss =  ce + 3 * ce1
        if step %20 == 0:
            print(epoch,i,ce.data.cpu().numpy(),ce1.data.cpu().numpy(),(torch.sum(torch.max(target_test_pred,dim = -1)[-1] == torch.max(target_val_label,dim = -1)[-1]) / 256.0).data.cpu().numpy())
        optimizer.zero_grad()
        optimizer1.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer1.step()
        
    train_loader.reset()
    train_loader1.reset()

torch.save(feature_extractor.state_dict(),'checkpoint/model_fe' +'.pth')
torch.save(cls.state_dict(),'checkpoint/model_cls' +'.pth')
