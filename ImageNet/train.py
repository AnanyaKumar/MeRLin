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
import  os
from    networks import *
from    utils import * 
from    merlin import * 
import cifar10_resized

train_transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
test_transform = transforms.ToTensor()
train_data = cifar10_resized.CIFAR10Resized(
    train_len=4096, train=True, root='/u/scr/ananya/cifar10_dataset',
    transform=train_transform)
test_data = cifar10_resized.CIFAR10Resized(
    train_len=4096, train=False, root='/u/scr/ananya/cifar10_dataset',
    transform=test_transform)

train_loader1 = torch.utils.data.DataLoader(
    train_data, batch_size=512, shuffle=True, num_workers=16)

feature_extractor = ResNetFc(model_name='resnet18',model_path='/data/pytorchModels/resnet18.pth').cuda()
# cls = CLS(feature_extractor.output_num(),1000, bottle_neck_dim=False).cuda()
feature_extractor = nn.DataParallel(feature_extractor)
# cls = nn.DataParallel(cls)

criterion = torch.nn.CrossEntropyLoss()
criterion1 = torch.nn.MSELoss()

optimizer = torch.optim.SGD(feature_extractor.parameters(),lr=1e-2,weight_decay=5e-4,momentum=0.9)
# optimizer1 = torch.optim.SGD(cls.parameters(),lr=1e-2,weight_decay=5e-4,momentum=0.9)

feature_extractor.train()
# cls.train()
step = 0

for epoch in range(200):
    if epoch % 100 == 1:
        torch.save(feature_extractor.state_dict(),'checkpoint/'+str(epoch) +'fe' +'.pth')
        # torch.save(cls.state_dict(),'checkpoint/'+str(epoch) +'cls' +'.pth')
    for i, data1 in enumerate(train_loader1):
        
        step = step + 1
        if epoch > 2400 & epoch <= 6400:
            for g in optimizer.param_groups:
                g['lr'] = 1e-3
           # for g in optimizer1.param_groups:
           #     g['lr'] = 1e-3
        if epoch > 6400:
            for g in optimizer.param_groups:
                g['lr'] = 1e-4
            #for g in optimizer1.param_groups:
            #    g['lr'] = 1e-4
        #image = data[0]["data"].cuda(non_blocking=True)
        #target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        image1 = data1[0].cuda(non_blocking=True)
        target1 = data1[1].squeeze().long().cuda(non_blocking=True)
        
        #fss,fs = cls(feature_extractor(image))
        ftt = feature_extractor(image1)
        
        target_label1, target_test_pred = merlin_linear(None,ftt,target1,256, 256, 10)
        
        ce1 = criterion(target_test_pred, target_label1)
        #ce = criterion(fs,target)
        
        loss =  ce1
        if step %20 == 0:
            print(epoch,i,ce1.data.cpu().numpy(),(torch.sum(torch.max(target_test_pred,dim = -1)[-1] == torch.max(target_label1,dim = -1)[-1]) / 256.0).data.cpu().numpy())
        optimizer.zero_grad()
        # optimizer1.zero_grad()
        loss.backward()
        optimizer.step()
        # optimizer1.step()
        
    # train_loader.reset()
    # train_loader1.reset()

torch.save(feature_extractor.state_dict(),'checkpoint/model_fe' +'.pth')
# torch.save(cls.state_dict(),'checkpoint/model_cls' +'.pth')
